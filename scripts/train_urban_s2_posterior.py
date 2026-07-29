from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from urban_s2.cells import (
    AdaptiveS2Partition,
    S2Tree,
    build_adaptive_s2_partition,
    build_s2_tree,
)
from urban_s2.geodesy import (
    EARTH_MEAN_RADIUS_KM,
    latlon_to_unit,
    unit_to_latlon,
)
from urban_s2.metrics import evaluate_point_predictions
from urban_s2.vmf import vmf_mixture_log_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the UrbanS2-Posterior pilot."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--snapshot-script", type=Path)
    return parser.parse_args()


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, dim=-1)


def torch_exp_map(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    base = normalize(base)
    tangent = tangent - (tangent * base).sum(dim=-1, keepdim=True) * base
    norm = tangent.norm(dim=-1, keepdim=True)
    direction = tangent / norm.clamp_min(1e-8)
    mapped = torch.cos(norm) * base + torch.sin(norm) * direction
    return normalize(torch.where(norm > 1e-8, mapped, base))


def bounded_tangent(
    raw: torch.Tensor,
    base: torch.Tensor,
    maximum_radians: float = 0.55,
) -> torch.Tensor:
    tangent = raw - (raw * base).sum(dim=-1, keepdim=True) * base
    norm = tangent.norm(dim=-1, keepdim=True)
    bounded_norm = maximum_radians * torch.tanh(norm)
    return tangent * (bounded_norm / norm.clamp_min(1e-8))


def log_vmf_normalizer(kappa: torch.Tensor) -> torch.Tensor:
    kappa = kappa.clamp_min(1e-4)
    log_sinh = (
        kappa
        - math.log(2.0)
        + torch.log1p(-torch.exp(-2.0 * kappa))
    )
    return torch.log(kappa) - math.log(4.0 * math.pi) - log_sinh


class HierarchicalRouter(nn.Module):
    def __init__(self, feature_dimension: int, tree: S2Tree) -> None:
        super().__init__()
        self.edge_logits = nn.Linear(
            feature_dimension,
            len(tree.node_parents) - 1,
        )
        self.register_buffer(
            "node_parents",
            torch.from_numpy(tree.node_parents).long(),
        )
        self.register_buffer(
            "node_levels",
            torch.from_numpy(tree.node_levels.astype(np.int64)),
        )
        self.register_buffer(
            "leaf_nodes",
            torch.from_numpy(tree.leaf_node_indices).long(),
        )
        self.minimum_level = int(tree.node_levels[1:].min())
        self.maximum_level = int(tree.node_levels.max())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.edge_logits(features)
        batch_size = raw.shape[0]
        node_count = len(self.node_parents)
        edge_parents = self.node_parents[1:]
        expanded_parents = edge_parents.unsqueeze(0).expand(batch_size, -1)

        maxima = torch.full(
            (batch_size, node_count),
            -torch.inf,
            dtype=raw.dtype,
            device=raw.device,
        )
        maxima.scatter_reduce_(
            1,
            expanded_parents,
            raw,
            reduce="amax",
            include_self=True,
        )
        centered = raw - maxima.gather(1, expanded_parents)
        sums = torch.zeros_like(maxima)
        sums.scatter_add_(1, expanded_parents, torch.exp(centered))
        log_normalizers = maxima + torch.log(sums.clamp_min(1e-20))
        edge_log_probabilities = raw - log_normalizers.gather(
            1,
            expanded_parents,
        )

        node_log_probabilities = torch.zeros(
            (batch_size, node_count),
            dtype=raw.dtype,
            device=raw.device,
        )
        for level in range(self.minimum_level, self.maximum_level + 1):
            nodes = torch.nonzero(self.node_levels == level).squeeze(1)
            if nodes.numel() == 0:
                continue
            parents = self.node_parents[nodes]
            node_log_probabilities[:, nodes] = (
                node_log_probabilities[:, parents]
                + edge_log_probabilities[:, nodes - 1]
            )
        return node_log_probabilities[:, self.leaf_nodes]


class UrbanS2Posterior(nn.Module):
    def __init__(
        self,
        *,
        feature_dimension: int,
        tree: S2Tree,
        prototypes: np.ndarray,
        centers_unit: np.ndarray,
        leaf_counts: np.ndarray,
    ) -> None:
        super().__init__()
        self.router = HierarchicalRouter(feature_dimension, tree)
        self.register_buffer(
            "prototypes",
            normalize(torch.from_numpy(prototypes).float()),
        )
        self.register_buffer(
            "centers_unit",
            normalize(torch.from_numpy(centers_unit).float()),
        )
        self.register_buffer(
            "log_leaf_counts",
            torch.log(torch.from_numpy(leaf_counts).float().clamp_min(1)),
        )
        hidden_dimension = 256
        self.offset_head = nn.Sequential(
            nn.Linear(feature_dimension * 2, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 3),
        )
        self.kappa_head = nn.Sequential(
            nn.Linear(feature_dimension * 2, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.density_logit = nn.Parameter(torch.tensor(-1.0))

    def fused_log_probabilities(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_log_probabilities = self.router(features)
        temperature = self.log_temperature.exp().clamp(0.01, 1.0)
        prototype_logits = features @ self.prototypes.T / temperature
        fusion_weight = 2.0 * torch.sigmoid(self.fusion_logit)
        density_weight = 2.0 * torch.sigmoid(self.density_logit)
        fused_logits = (
            router_log_probabilities
            + fusion_weight * prototype_logits
            - density_weight * self.log_leaf_counts
        )
        return (
            torch.log_softmax(fused_logits, dim=1),
            router_log_probabilities,
        )

    def component_parameters(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = self.prototypes[labels]
        centers = self.centers_unit[labels]
        inputs = torch.cat((features, prototypes), dim=-1)
        tangent = bounded_tangent(self.offset_head(inputs), centers)
        means = torch_exp_map(centers, tangent)
        kappas = (
            torch.nn.functional.softplus(self.kappa_head(inputs).squeeze(-1))
            + 1e-3
        ).clamp_max(500.0)
        return means, kappas

    def posterior(
        self,
        features: torch.Tensor,
        top_k: int,
    ) -> dict[str, torch.Tensor]:
        log_probabilities, router_log_probabilities = (
            self.fused_log_probabilities(features)
        )
        top_k = min(top_k, log_probabilities.shape[1])
        top_log_probabilities, labels = torch.topk(
            log_probabilities,
            k=top_k,
            dim=1,
        )
        weights = torch.softmax(top_log_probabilities, dim=1)
        expanded_features = features[:, None, :].expand(-1, top_k, -1)
        prototypes = self.prototypes[labels]
        centers = self.centers_unit[labels]
        inputs = torch.cat((expanded_features, prototypes), dim=-1)
        tangent = bounded_tangent(self.offset_head(inputs), centers)
        means = torch_exp_map(centers, tangent)
        kappas = (
            torch.nn.functional.softplus(self.kappa_head(inputs).squeeze(-1))
            + 1e-3
        ).clamp_max(500.0)
        resultant_lengths = (
            1.0 / torch.tanh(kappas.clamp_min(1e-3))
            - 1.0 / kappas.clamp_min(1e-3)
        ).clamp_min(0.0)
        point = normalize(
            (weights[..., None] * resultant_lengths[..., None] * means).sum(
                dim=1
            )
        )
        router_label = router_log_probabilities.argmax(dim=1)
        prototype_label = (features @ self.prototypes.T).argmax(dim=1)
        return {
            "point": point,
            "labels": labels,
            "weights": weights,
            "means": means,
            "kappas": kappas,
            "router_label": router_label,
            "prototype_label": prototype_label,
        }


def build_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    number_of_leaves: int,
) -> np.ndarray:
    prototypes = np.zeros(
        (number_of_leaves, features.shape[1]),
        dtype=np.float64,
    )
    np.add.at(prototypes, labels, features.astype(np.float64))
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("At least one S2 leaf has an empty prototype.")
    return (prototypes / norms).astype(np.float32)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def summarize_metrics(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    metrics = evaluate_point_predictions(ground_truth, predictions)
    errors = np.asarray(metrics.pop("errors_km"))
    metrics["geoscore"] = float(np.mean(np.exp(-errors / 1492.7)))
    return metrics, errors


def nearest_retrieval(
    query_features: torch.Tensor,
    train_features: torch.Tensor,
    train_coordinates: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(query_features), batch_size):
        query = query_features[start : start + batch_size]
        indices = (query @ train_features.T).argmax(dim=1).cpu().numpy()
        predictions.append(train_coordinates[indices])
    return np.concatenate(predictions, axis=0)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["pilot"]["seeds"][0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    features = np.load(args.features, mmap_mode="r").astype(np.float32)
    feature_index = pd.read_parquet(args.feature_index)
    if len(features) != len(feature_index):
        raise RuntimeError("Feature matrix and feature index length differ.")
    if not np.isfinite(features).all():
        raise RuntimeError("Features contain non-finite values.")

    train_mask = feature_index["split"].eq("train").to_numpy()
    train_features_np = features[train_mask]
    train_index = feature_index.loc[train_mask].reset_index(drop=True)
    partition = build_adaptive_s2_partition(
        train_index["latitude"].to_numpy(),
        train_index["longitude"].to_numpy(),
        min_level=int(config["model"]["s2_min_level"]),
        max_level=int(config["model"]["s2_max_level"]),
        max_samples_per_leaf=int(
            config["model"]["maximum_samples_per_leaf"]
        ),
    )
    tree = build_s2_tree(
        partition.leaf_cell_ids,
        partition.leaf_levels,
        min_level=int(config["model"]["s2_min_level"]),
    )
    prototypes = build_prototypes(
        train_features_np,
        partition.labels,
        len(partition.leaf_counts),
    )

    device = torch.device("cuda")
    model = UrbanS2Posterior(
        feature_dimension=features.shape[1],
        tree=tree,
        prototypes=prototypes,
        centers_unit=partition.leaf_centers_unit,
        leaf_counts=partition.leaf_counts,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    train_features = torch.from_numpy(train_features_np).float()
    train_labels = torch.from_numpy(partition.labels).long()
    train_units = torch.from_numpy(
        latlon_to_unit(
            train_index["latitude"].to_numpy(),
            train_index["longitude"].to_numpy(),
        ).astype(np.float32)
    )
    loader = DataLoader(
        TensorDataset(train_features, train_labels, train_units),
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    last_checkpoint = args.checkpoint_root / "last.pt"
    start_epoch = 0
    best_development_median = math.inf
    if last_checkpoint.is_file():
        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_development_median = float(
            checkpoint["best_development_median_km"]
        )

    centers = model.centers_unit
    epochs = int(config["pilot"]["epochs"])
    epoch_records: list[dict[str, float]] = []
    training_started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sums: dict[str, float] = {
            "total": 0.0,
            "router": 0.0,
            "geo_soft": 0.0,
            "offset": 0.0,
            "vmf": 0.0,
        }
        sample_count = 0
        epoch_started = time.perf_counter()
        for batch_features, labels, targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            fused_log_probabilities, router_log_probabilities = (
                model.fused_log_probabilities(batch_features)
            )
            router_loss = -router_log_probabilities[
                torch.arange(len(labels), device=device),
                labels,
            ].mean()

            center_dots = torch.clamp(
                targets @ centers.T,
                -1.0 + 1e-6,
                1.0 - 1e-6,
            )
            distances_km = (
                torch.acos(center_dots) * EARTH_MEAN_RADIUS_KM
            )
            soft_targets = torch.softmax(
                -distances_km
                / float(config["model"]["geo_soft_radius_km"]),
                dim=1,
            )
            geo_soft_loss = -(
                soft_targets * fused_log_probabilities
            ).sum(dim=1).mean()

            means, kappas = model.component_parameters(batch_features, labels)
            dots = torch.clamp(
                (means * targets).sum(dim=1),
                -1.0 + 1e-6,
                1.0 - 1e-6,
            )
            offset_loss = torch.acos(dots).mean()
            vmf_loss = -(log_vmf_normalizer(kappas) + kappas * dots).mean()
            total_loss = (
                0.5 * router_loss
                + geo_soft_loss
                + 2.0 * offset_loss
                + 0.05 * vmf_loss
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite training loss.")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_count = len(labels)
            sample_count += batch_count
            loss_sums["total"] += float(total_loss.detach()) * batch_count
            loss_sums["router"] += float(router_loss.detach()) * batch_count
            loss_sums["geo_soft"] += float(geo_soft_loss.detach()) * batch_count
            loss_sums["offset"] += float(offset_loss.detach()) * batch_count
            loss_sums["vmf"] += float(vmf_loss.detach()) * batch_count

        development_mask = feature_index["split"].eq("development").to_numpy()
        development_features = torch.from_numpy(
            features[development_mask]
        ).float().to(device)
        with torch.inference_mode():
            development_point = model.posterior(
                development_features,
                int(config["model"]["posterior_top_k"]),
            )["point"].cpu().numpy()
        development_latlon = unit_to_latlon(development_point)
        development_truth = feature_index.loc[
            development_mask,
            ["latitude", "longitude"],
        ].to_numpy()
        development_metrics, _ = summarize_metrics(
            development_truth,
            development_latlon,
        )
        development_median = float(development_metrics["median_error_km"])
        record = {
            "epoch": float(epoch),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            **{
                f"{name}_loss": total / sample_count
                for name, total in loss_sums.items()
            },
            "development_median_error_km": development_median,
        }
        epoch_records.append(record)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_development_median_km": min(
                best_development_median,
                development_median,
            ),
            "partition": {
                "leaf_cell_ids": partition.leaf_cell_ids,
                "leaf_levels": partition.leaf_levels,
                "leaf_counts": partition.leaf_counts,
                "leaf_centers_unit": partition.leaf_centers_unit,
                "leaf_tokens": partition.leaf_tokens,
            },
            "tree": {
                "node_cell_ids": tree.node_cell_ids,
                "node_levels": tree.node_levels,
                "node_parents": tree.node_parents,
                "leaf_node_indices": tree.leaf_node_indices,
            },
            "config": config,
            "epoch_record": record,
        }
        atomic_torch_save(
            checkpoint_payload,
            args.checkpoint_root / f"epoch_{epoch:03d}.pt",
        )
        atomic_torch_save(checkpoint_payload, last_checkpoint)
        if development_median < best_development_median:
            best_development_median = development_median
            atomic_torch_save(
                checkpoint_payload,
                args.checkpoint_root / "best.pt",
            )
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.snapshot_script:
            subprocess.run(
                [str(args.snapshot_script), f"pilot_1pct_epoch_{epoch:03d}"],
                check=True,
            )

    training_seconds = time.perf_counter() - training_started
    best_checkpoint = torch.load(
        args.checkpoint_root / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()

    train_features_device = train_features.to(device)
    train_coordinates = train_index[["latitude", "longitude"]].to_numpy()
    prediction_frames: list[pd.DataFrame] = []
    all_metrics: dict[str, Any] = {}
    evaluation_started = time.perf_counter()
    for split in ("development", "official_test_pilot"):
        mask = feature_index["split"].eq(split).to_numpy()
        query_index = feature_index.loc[mask].reset_index(drop=True)
        query_features = torch.from_numpy(features[mask]).float().to(device)
        truth = query_index[["latitude", "longitude"]].to_numpy()
        with torch.inference_mode():
            posterior = model.posterior(
                query_features,
                int(config["model"]["posterior_top_k"]),
            )
        point_latlon = unit_to_latlon(posterior["point"].cpu().numpy())
        router_latlon = unit_to_latlon(
            model.centers_unit[posterior["router_label"]].cpu().numpy()
        )
        prototype_latlon = unit_to_latlon(
            model.centers_unit[posterior["prototype_label"]].cpu().numpy()
        )
        retrieval_latlon = nearest_retrieval(
            query_features,
            train_features_device,
            train_coordinates,
        )

        method_predictions = {
            "streetclip_retrieval": retrieval_latlon,
            "prototype_top1": prototype_latlon,
            "hierarchical_router": router_latlon,
            "urban_s2_posterior": point_latlon,
        }
        split_metrics: dict[str, Any] = {}
        errors_by_method: dict[str, np.ndarray] = {}
        for method, predictions in method_predictions.items():
            metrics, errors = summarize_metrics(truth, predictions)
            split_metrics[method] = metrics
            errors_by_method[method] = errors

        means = posterior["means"].cpu().numpy()
        kappas = posterior["kappas"].cpu().numpy()
        weights = posterior["weights"].cpu().numpy()
        target_unit = latlon_to_unit(truth[:, 0], truth[:, 1])
        split_metrics["urban_s2_posterior"]["mean_vmf_nll"] = float(
            -np.mean(
                vmf_mixture_log_prob(
                    target_unit,
                    means,
                    kappas,
                    weights,
                )
            )
        )
        all_metrics[split] = split_metrics

        frame = query_index[
            [
                "id",
                "sequence",
                "country",
                "latitude",
                "longitude",
            ]
        ].copy()
        frame.insert(0, "evaluation_split", split)
        for method, predictions in method_predictions.items():
            frame[f"{method}_latitude"] = predictions[:, 0]
            frame[f"{method}_longitude"] = predictions[:, 1]
            frame[f"{method}_error_km"] = errors_by_method[method]
        candidate_latlon = unit_to_latlon(means)
        frame["candidate_labels"] = posterior["labels"].cpu().numpy().tolist()
        frame["candidate_weights"] = weights.tolist()
        frame["candidate_kappas"] = kappas.tolist()
        frame["candidate_latitudes"] = candidate_latlon[:, :, 0].tolist()
        frame["candidate_longitudes"] = candidate_latlon[:, :, 1].tolist()
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_predictions = args.predictions_output.with_suffix(
        args.predictions_output.suffix + ".partial"
    )
    predictions.to_parquet(temporary_predictions, index=False)
    temporary_predictions.replace(args.predictions_output)

    metrics_payload = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_development_median_km": best_development_median,
        "s2_leaf_count": len(partition.leaf_counts),
        "s2_tree_node_count": len(tree.node_parents),
        "leaf_count_summary": {
            "minimum": int(partition.leaf_counts.min()),
            "median": float(np.median(partition.leaf_counts)),
            "maximum": int(partition.leaf_counts.max()),
        },
        "metrics": all_metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    timing_payload = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "train_images": int(train_mask.sum()),
        "test_images": int(
            feature_index["split"].eq("official_test_pilot").sum()
        ),
        "preprocessing_seconds": 0.0,
        "feature_seconds": 0.0,
        "train_seconds": training_seconds,
        "train_epochs": epochs,
        "train_seeds": 1,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "epoch_records": epoch_records,
    }
    args.timing_output.parent.mkdir(parents=True, exist_ok=True)
    args.timing_output.write_text(
        json.dumps(timing_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics_payload, indent=2, sort_keys=True))
    if args.snapshot_script:
        subprocess.run(
            [str(args.snapshot_script), "pilot_1pct_training_complete"],
            check=True,
        )


if __name__ == "__main__":
    main()
