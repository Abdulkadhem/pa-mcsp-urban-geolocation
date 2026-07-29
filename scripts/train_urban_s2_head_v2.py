"""Run the local, head-only StreetCLIP v2 diagnostic.

The script never evaluates ``official_test_pilot``.  It reuses frozen features,
builds train-only S2 prototypes, runs three seeded residual-head ablations, and
writes development-only metrics, predictions, checkpoints, and profiling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from urban_s2.cells import build_adaptive_s2_partition
from urban_s2.geodesy import (
    EARTH_MEAN_RADIUS_KM,
    latlon_to_unit,
    unit_to_latlon,
)
from urban_s2.head_v2 import PrototypeResidualHeadV2, normalize
from urban_s2.metrics import evaluate_point_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the development-only StreetCLIP head v2."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional subset of configured seeds for sequential screening.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional subset of configured variant names.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        raise RuntimeError("An S2 leaf has an empty feature prototype.")
    return (prototypes / norms).astype(np.float32)


def metrics_from_points(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    metrics = evaluate_point_predictions(truth, predictions)
    errors = np.asarray(metrics.pop("errors_km"), dtype=np.float64)
    metrics["geoscore"] = float(np.mean(np.exp(-errors / 1492.7)))
    return metrics, errors


def candidate_errors_km(
    truth_unit: np.ndarray,
    candidate_unit: np.ndarray,
) -> np.ndarray:
    dots = np.einsum("nd,nkd->nk", truth_unit, candidate_unit)
    return (
        np.arccos(np.clip(dots, -1.0, 1.0)) * EARTH_MEAN_RADIUS_KM
    )


def calibration_summary(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (
                confidence <= edges[index + 1]
            )
        else:
            selected = (confidence >= edges[index]) & (
                confidence < edges[index + 1]
            )
        if selected.any():
            ece += (
                selected.mean()
                * abs(confidence[selected].mean() - correct[selected].mean())
            )
    return {
        "ece_10": float(ece),
        "brier_25km": float(np.mean((confidence - correct) ** 2)),
        "mean_confidence": float(confidence.mean()),
        "accuracy_25km": float(correct.mean()),
    }


def cluster_bootstrap_accuracy_difference(
    proposed_correct: np.ndarray,
    baseline_correct: np.ndarray,
    clusters: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    proposed_correct = np.asarray(proposed_correct, dtype=np.float64)
    baseline_correct = np.asarray(baseline_correct, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    cluster_differences = np.zeros(len(unique_clusters), dtype=np.float64)
    cluster_sizes = np.zeros(len(unique_clusters), dtype=np.float64)
    np.add.at(
        cluster_differences,
        inverse,
        proposed_correct - baseline_correct,
    )
    np.add.at(cluster_sizes, inverse, 1.0)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = generator.integers(
            0,
            len(unique_clusters),
            size=len(unique_clusters),
        )
        draws[index] = (
            cluster_differences[sampled].sum()
            / cluster_sizes[sampled].sum()
        )
    observed = float(
        np.mean(proposed_correct - baseline_correct)
    )
    return {
        "difference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def cluster_bootstrap_mean_log_error_difference(
    proposed_errors: np.ndarray,
    baseline_errors: np.ndarray,
    clusters: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap proposed-minus-baseline mean log1p(error) by cluster."""
    proposed_errors = np.asarray(proposed_errors, dtype=np.float64)
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    differences = np.log1p(proposed_errors) - np.log1p(baseline_errors)
    clusters = np.asarray(clusters)
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    cluster_sums = np.zeros(len(unique_clusters), dtype=np.float64)
    cluster_sizes = np.zeros(len(unique_clusters), dtype=np.float64)
    np.add.at(cluster_sums, inverse, differences)
    np.add.at(cluster_sizes, inverse, 1.0)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = generator.integers(
            0,
            len(unique_clusters),
            size=len(unique_clusters),
        )
        draws[index] = (
            cluster_sums[sampled].sum()
            / cluster_sizes[sampled].sum()
        )
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "direction": "negative_favors_proposed",
    }


def gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def current_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except ImportError:
        return None


def ram_usage_fraction() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent / 100.0)
    except ImportError:
        return None


def enforce_ram_limit(maximum_fraction: float = 0.70) -> None:
    usage = ram_usage_fraction()
    if usage is not None and usage >= maximum_fraction:
        raise RuntimeError(
            f"RAM guard stopped v2 at {usage:.1%}; "
            f"the configured limit is {maximum_fraction:.1%}."
        )


def batched_anchor_logits(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    temperature: float,
    batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    prototype_transpose = normalize(prototypes).T
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = normalize(features[start : start + batch_size])
            output.append(
                (batch @ prototype_transpose / temperature).cpu()
            )
    return torch.cat(output, dim=0)


def nearest_retrieval(
    query_features: torch.Tensor,
    train_features: torch.Tensor,
    train_coordinates: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    normalized_train = normalize(train_features)
    with torch.inference_mode():
        for start in range(0, len(query_features), batch_size):
            query = normalize(query_features[start : start + batch_size])
            indices = (
                query @ normalized_train.T
            ).argmax(dim=1).cpu().numpy()
            predictions.append(train_coordinates[indices])
    return np.concatenate(predictions, axis=0)


def evaluate_model(
    model: PrototypeResidualHeadV2,
    features: torch.Tensor,
    anchor_logits: torch.Tensor,
    truth_latlon: np.ndarray,
    truth_unit: np.ndarray,
    *,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    map_points: list[np.ndarray] = []
    mean_points: list[np.ndarray] = []
    candidate_points: list[np.ndarray] = []
    candidate_weights: list[np.ndarray] = []
    candidate_labels: list[np.ndarray] = []
    candidate_kappas: list[np.ndarray] = []
    candidate_gates: list[np.ndarray] = []
    residual_magnitudes: list[np.ndarray] = []
    model.eval()
    evaluation_started = time.perf_counter()
    for start in range(0, len(features), batch_size):
        batch_features = features[start : start + batch_size].to(device)
        batch_anchor = anchor_logits[start : start + batch_size].to(device)
        result = model.posterior(
            batch_features,
            top_k=top_k,
            anchor_logits=batch_anchor,
        )
        map_points.append(result["map_point"].cpu().numpy())
        mean_points.append(result["global_mean"].cpu().numpy())
        candidate_points.append(result["means"].cpu().numpy())
        candidate_weights.append(result["weights"].cpu().numpy())
        candidate_labels.append(result["labels"].cpu().numpy())
        candidate_kappas.append(result["kappas"].cpu().numpy())
        candidate_gates.append(result["gates"].cpu().numpy())
        residual_magnitudes.append(
            result["residual_logits"].abs().mean(dim=1).cpu().numpy()
        )
    map_unit = np.concatenate(map_points)
    mean_unit = np.concatenate(mean_points)
    candidates = np.concatenate(candidate_points)
    weights = np.concatenate(candidate_weights)
    labels = np.concatenate(candidate_labels)
    kappas = np.concatenate(candidate_kappas)
    gates = np.concatenate(candidate_gates)
    residual_abs = np.concatenate(residual_magnitudes)
    map_latlon = unit_to_latlon(map_unit)
    mean_latlon = unit_to_latlon(mean_unit)
    map_metrics, map_errors = metrics_from_points(
        truth_latlon,
        map_latlon,
    )
    mean_metrics, mean_errors = metrics_from_points(
        truth_latlon,
        mean_latlon,
    )
    oracle_errors = candidate_errors_km(truth_unit, candidates).min(axis=1)
    entropy = -(
        weights * np.log(np.clip(weights, 1e-12, 1.0))
    ).sum(axis=1)
    max_weight = weights.max(axis=1)
    calibration = calibration_summary(
        max_weight,
        map_errors <= 25.0,
    )
    profile = {
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "candidate_oracle": {
            "median_error_km": float(np.median(oracle_errors)),
            "accuracy_1km": float(np.mean(oracle_errors <= 1.0)),
            "accuracy_25km": float(np.mean(oracle_errors <= 25.0)),
            "accuracy_200km": float(np.mean(oracle_errors <= 200.0)),
        },
        "posterior": {
            "mean_max_weight": float(max_weight.mean()),
            "median_max_weight": float(np.median(max_weight)),
            "mean_entropy_nats": float(entropy.mean()),
            "effective_components": float(np.exp(entropy.mean())),
            "median_kappa": float(np.median(kappas)),
            "median_gate": float(np.median(gates)),
            "mean_abs_residual_logit": float(residual_abs.mean()),
        },
        "calibration": calibration,
    }
    metrics = {
        "map": map_metrics,
        "global_mean_ablation": mean_metrics,
        "profiling": profile,
    }
    arrays = {
        "map_latlon": map_latlon,
        "map_errors": map_errors,
        "mean_latlon": mean_latlon,
        "mean_errors": mean_errors,
        "oracle_errors": oracle_errors,
        "max_weight": max_weight,
        "entropy": entropy,
        "labels": labels,
    }
    return metrics, arrays


def train_variant(
    *,
    seed: int,
    variant: dict[str, Any],
    config: dict[str, Any],
    prototypes: torch.Tensor,
    centers: torch.Tensor,
    leaf_counts: torch.Tensor,
    train_features: torch.Tensor,
    train_anchor_logits: torch.Tensor,
    train_labels: torch.Tensor,
    train_units: torch.Tensor,
    development_features: torch.Tensor,
    development_anchor_logits: torch.Tensor,
    development_truth: np.ndarray,
    development_units: np.ndarray,
    device: torch.device,
) -> tuple[PrototypeResidualHeadV2, dict[str, Any], dict[str, np.ndarray]]:
    set_seed(seed)
    model_config = config["model"]
    training_config = config["training"]
    model = PrototypeResidualHeadV2(
        feature_dimension=train_features.shape[1],
        prototypes=prototypes,
        centers_unit=centers,
        leaf_counts=leaf_counts,
        hidden_dimension=int(model_config["hidden_dimension"]),
        temperature=float(model_config["temperature"]),
        density_weight=float(variant.get("density_weight", 0.0)),
        maximum_offset_radians=float(
            model_config["maximum_offset_radians"]
        ),
        enable_offsets=bool(variant["enable_offsets"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(
            train_features,
            train_anchor_logits,
            train_labels,
            train_units,
        ),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    epoch_records: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    peak_rss = current_rss_mb()
    for epoch in range(int(training_config["epochs"])):
        model.train()
        sums = {
            "total": 0.0,
            "listwise": 0.0,
            "mixture_nll": 0.0,
            "offset": 0.0,
            "gate": 0.0,
            "anchor": 0.0,
            "natural_topk_recall": 0.0,
            "forced_inclusion_rate": 0.0,
        }
        count = 0
        first_gradient_record: dict[str, float] | None = None
        epoch_started = time.perf_counter()
        for (
            batch_features,
            batch_anchor,
            batch_labels,
            batch_targets,
        ) in loader:
            batch_features = batch_features.to(device)
            batch_anchor = batch_anchor.to(device)
            batch_labels = batch_labels.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                batch_features,
                batch_labels,
                batch_targets,
                top_k=int(model_config["top_k"]),
                geo_scale_km=float(training_config["geo_scale_km"]),
                near_radius_km=float(
                    training_config["near_radius_km"]
                ),
                mixture_weight=float(
                    variant.get(
                        "mixture_weight",
                        training_config["mixture_weight"],
                    )
                ),
                offset_weight=float(
                    training_config["offset_weight"]
                    if variant["enable_offsets"]
                    else 0.0
                ),
                gate_weight=float(
                    training_config["gate_weight"]
                    if variant["enable_offsets"]
                    else 0.0
                ),
                anchor_weight=float(training_config["anchor_weight"]),
                anchor_logits=batch_anchor,
            )
            if not torch.isfinite(loss.total):
                raise RuntimeError("The v2 head produced a non-finite loss.")
            loss.total.backward()
            if first_gradient_record is None:
                first_gradient_record = {
                    "residual_scorer": gradient_norm(
                        model.residual_scorer
                    ),
                    "offset_head": gradient_norm(model.offset_head),
                    "offset_gate_head": gradient_norm(
                        model.offset_gate_head
                    ),
                    "kappa_head": gradient_norm(model.kappa_head),
                }
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training_config["gradient_clip_norm"]),
            )
            optimizer.step()
            batch_count = len(batch_labels)
            count += batch_count
            values = vars(loss)
            for name in sums:
                value = values[name]
                sums[name] += float(value.detach()) * batch_count
        rss = current_rss_mb()
        enforce_ram_limit()
        if rss is not None:
            peak_rss = max(peak_rss or 0.0, rss)
        epoch_records.append(
            {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "samples_per_second": (
                    count / (time.perf_counter() - epoch_started)
                ),
                "losses": {
                    name: value / count for name, value in sums.items()
                },
                "first_batch_gradient_norms": first_gradient_record,
                "rss_mb": rss,
            }
        )

    metrics, arrays = evaluate_model(
        model,
        development_features,
        development_anchor_logits,
        development_truth,
        development_units,
        top_k=int(model_config["top_k"]),
        batch_size=int(training_config["batch_size"]),
        device=device,
    )
    metrics["training"] = {
        "seconds": time.perf_counter() - training_started,
        "epochs": int(training_config["epochs"]),
        "peak_rss_mb": peak_rss,
        "epoch_records": epoch_records,
    }
    return model, metrics, arrays


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["development_split"] != "development":
        raise RuntimeError("v2 is restricted to the development split.")
    if "official_test_pilot" not in config["forbidden_evaluation_splits"]:
        raise RuntimeError("The official pilot test must remain forbidden.")
    device = choose_device(args.device)
    configured_variants = {
        variant["name"]: variant for variant in config["variants"]
    }
    selected_seed_values = (
        [int(seed) for seed in args.seeds]
        if args.seeds
        else [int(seed) for seed in config["seeds"]]
    )
    selected_variant_names = (
        list(args.variants)
        if args.variants
        else list(configured_variants)
    )
    unknown_variants = sorted(
        set(selected_variant_names) - set(configured_variants)
    )
    if unknown_variants:
        raise ValueError(f"Unknown v2 variants: {unknown_variants}")
    selected_variants = [
        configured_variants[name] for name in selected_variant_names
    ]
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_float32_matmul_precision("high")
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    tracemalloc.start()
    enforce_ram_limit()
    run_started = time.perf_counter()
    stage_times: dict[str, float] = {}

    stage = time.perf_counter()
    features = np.load(args.features, mmap_mode="r").astype(np.float32)
    feature_index = pd.read_parquet(args.feature_index)
    if len(features) != len(feature_index):
        raise RuntimeError("Feature matrix and index length differ.")
    if not np.isfinite(features).all():
        raise RuntimeError("Features contain non-finite values.")
    stage_times["load_seconds"] = time.perf_counter() - stage

    train_mask = feature_index["split"].eq("train").to_numpy()
    development_mask = feature_index["split"].eq("development").to_numpy()
    evaluated_splits = set(
        feature_index.loc[development_mask, "split"].astype(str)
    )
    if evaluated_splits != {"development"}:
        raise RuntimeError("A forbidden split entered v2 evaluation.")
    train_index = feature_index.loc[train_mask].reset_index(drop=True)
    development_index = feature_index.loc[
        development_mask
    ].reset_index(drop=True)

    stage = time.perf_counter()
    partition_config = config["partition"]
    partition = build_adaptive_s2_partition(
        train_index["latitude"].to_numpy(),
        train_index["longitude"].to_numpy(),
        min_level=int(partition_config["s2_min_level"]),
        max_level=int(partition_config["s2_max_level"]),
        max_samples_per_leaf=int(
            partition_config["maximum_samples_per_leaf"]
        ),
    )
    train_features_np = features[train_mask]
    prototypes_np = build_prototypes(
        train_features_np,
        partition.labels,
        len(partition.leaf_counts),
    )
    stage_times["partition_and_prototypes_seconds"] = (
        time.perf_counter() - stage
    )

    train_features = torch.from_numpy(train_features_np).float()
    development_features = torch.from_numpy(
        features[development_mask]
    ).float()
    prototypes = torch.from_numpy(prototypes_np).float()
    centers = torch.from_numpy(
        partition.leaf_centers_unit.astype(np.float32)
    )
    leaf_counts = torch.from_numpy(partition.leaf_counts).long()
    train_labels = torch.from_numpy(partition.labels).long()
    train_units_np = latlon_to_unit(
        train_index["latitude"].to_numpy(),
        train_index["longitude"].to_numpy(),
    ).astype(np.float32)
    train_units = torch.from_numpy(train_units_np)
    development_truth = development_index[
        ["latitude", "longitude"]
    ].to_numpy()
    development_units = latlon_to_unit(
        development_truth[:, 0],
        development_truth[:, 1],
    )

    stage = time.perf_counter()
    anchor_batch_size = max(
        1024,
        int(config["training"]["batch_size"]),
    )
    train_anchor_logits = batched_anchor_logits(
        train_features,
        prototypes,
        temperature=float(config["model"]["temperature"]),
        batch_size=anchor_batch_size,
    )
    development_anchor_logits = batched_anchor_logits(
        development_features,
        prototypes,
        temperature=float(config["model"]["temperature"]),
        batch_size=anchor_batch_size,
    )
    stage_times["anchor_logit_cache_seconds"] = time.perf_counter() - stage

    anchor_labels = development_anchor_logits.argmax(dim=1).numpy()
    anchor_latlon = unit_to_latlon(
        partition.leaf_centers_unit[anchor_labels]
    )
    anchor_metrics, anchor_errors = metrics_from_points(
        development_truth,
        anchor_latlon,
    )
    stage = time.perf_counter()
    retrieval_latlon = nearest_retrieval(
        development_features,
        train_features,
        train_index[["latitude", "longitude"]].to_numpy(),
        batch_size=anchor_batch_size,
    )
    retrieval_metrics, retrieval_errors = metrics_from_points(
        development_truth,
        retrieval_latlon,
    )
    stage_times["retrieval_baseline_seconds"] = time.perf_counter() - stage

    all_results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    bootstrap_resamples = int(config["gate"]["bootstrap_resamples"])
    for variant in selected_variants:
        variant_results: dict[str, Any] = {}
        for seed in selected_seed_values:
            model, metrics, arrays = train_variant(
                seed=int(seed),
                variant=variant,
                config=config,
                prototypes=prototypes,
                centers=centers,
                leaf_counts=leaf_counts,
                train_features=train_features,
                train_anchor_logits=train_anchor_logits,
                train_labels=train_labels,
                train_units=train_units,
                development_features=development_features,
                development_anchor_logits=development_anchor_logits,
                development_truth=development_truth,
                development_units=development_units,
                device=device,
            )
            bootstrap = cluster_bootstrap_accuracy_difference(
                arrays["map_errors"] <= 25.0,
                retrieval_errors <= 25.0,
                development_index["sequence"].astype(str).to_numpy(),
                resamples=bootstrap_resamples,
                seed=int(seed),
            )
            metrics["paired_cluster_bootstrap_vs_retrieval_25km"] = (
                bootstrap
            )
            metrics[
                "paired_cluster_bootstrap_log1p_error_vs_prototype"
            ] = cluster_bootstrap_mean_log_error_difference(
                arrays["map_errors"],
                anchor_errors,
                development_index["sequence"].astype(str).to_numpy(),
                resamples=bootstrap_resamples,
                seed=int(seed) + 10_000,
            )
            variant_results[str(seed)] = metrics
            torch.save(
                {
                    "schema_version": 2,
                    "seed": int(seed),
                    "variant": variant,
                    "model_state": model.state_dict(),
                    "partition": {
                        "leaf_cell_ids": partition.leaf_cell_ids,
                        "leaf_levels": partition.leaf_levels,
                        "leaf_counts": partition.leaf_counts,
                        "leaf_centers_unit": partition.leaf_centers_unit,
                        "leaf_tokens": partition.leaf_tokens,
                    },
                    "config": config,
                },
                checkpoint_root
                / f"{variant['name']}_seed_{int(seed)}.pt",
            )
            frame = development_index[
                [
                    "id",
                    "sequence",
                    "country",
                    "latitude",
                    "longitude",
                ]
            ].copy()
            frame.insert(0, "seed", int(seed))
            frame.insert(0, "variant", variant["name"])
            frame.insert(0, "evaluation_split", "development")
            frame["map_latitude"] = arrays["map_latlon"][:, 0]
            frame["map_longitude"] = arrays["map_latlon"][:, 1]
            frame["map_error_km"] = arrays["map_errors"]
            frame["global_mean_error_km"] = arrays["mean_errors"]
            frame["oracle_topk_error_km"] = arrays["oracle_errors"]
            frame["max_weight"] = arrays["max_weight"]
            frame["entropy_nats"] = arrays["entropy"]
            prediction_frames.append(frame)
        all_results[variant["name"]] = variant_results

    gate_config = config["gate"]
    full_results = all_results.get("full_v2", {})
    seed_gate_results: dict[str, bool] = {}
    for seed, result in full_results.items():
        map_metrics = result["map"]
        log_error_bootstrap = result[
            "paired_cluster_bootstrap_log1p_error_vs_prototype"
        ]
        seed_gate_results[seed] = bool(
            map_metrics["accuracy"]["25"]
            >= retrieval_metrics["accuracy"]["25"]
            + float(gate_config["accuracy_margin_over_retrieval"])
            and map_metrics["median_error_km"]
            <= anchor_metrics["median_error_km"]
            * float(
                gate_config["maximum_median_ratio_to_prototype"]
            )
            and map_metrics["geoscore"] >= anchor_metrics["geoscore"]
            and log_error_bootstrap["ci95_high"] < 0.0
        )
    required_seed_keys = {str(seed) for seed in config["seeds"]}
    final_gate_evaluated = (
        set(seed_gate_results) == required_seed_keys
        and "full_v2" in selected_variant_names
    )
    final_gate_passed = bool(
        final_gate_evaluated
        and all(seed_gate_results.values())
    )
    screening_candidate_eligible = bool(
        seed_gate_results and all(seed_gate_results.values())
    )
    accuracy_bootstrap = {
        seed: result[
            "paired_cluster_bootstrap_vs_retrieval_25km"
        ]
        for seed, result in full_results.items()
    }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = (
        args.output_root / "head_v2_development_predictions.parquet"
    )
    predictions.to_parquet(predictions_path, index=False)
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = {
        "schema_version": 2,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "official_test_evaluated": False,
        "evaluated_splits": ["development"],
        "counts": {
            "train": int(train_mask.sum()),
            "development": int(development_mask.sum()),
            "s2_leaves": int(len(partition.leaf_counts)),
        },
        "screening_selection": {
            "seeds": selected_seed_values,
            "variants": selected_variant_names,
            "sequential_execution": True,
        },
        "baselines": {
            "prototype_anchor": anchor_metrics,
            "streetclip_retrieval": retrieval_metrics,
        },
        "variants": all_results,
        "profiling": {
            "total_seconds": time.perf_counter() - run_started,
            "stage_seconds": stage_times,
            "python_current_memory_mb": current_memory / (1024**2),
            "python_peak_memory_mb": peak_memory / (1024**2),
            "process_rss_mb": current_rss_mb(),
            "torch_threads": torch.get_num_threads(),
            "ram_usage_fraction": ram_usage_fraction(),
            "ram_limit_fraction": 0.70,
        },
        "gate": {
            "criteria": gate_config,
            "seed_passes": seed_gate_results,
            "accuracy_bootstrap_secondary": accuracy_bootstrap,
            "evaluated": final_gate_evaluated,
            "passed": final_gate_passed,
            "screening_candidate_eligible": screening_candidate_eligible,
            "decision": (
                "ACCEPT_CANDIDATE"
                if final_gate_passed
                else (
                    "REJECT_CANDIDATE"
                    if final_gate_evaluated
                    else (
                        "PROMOTE_TO_3_SEEDS"
                        if screening_candidate_eligible
                        else "STOP_SCREENING"
                    )
                )
            ),
        },
    }
    metrics_path = args.output_root / "head_v2_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
