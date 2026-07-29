#!/usr/bin/env python3
"""Train and screen the safe PA-MCSP reranker on frozen 10% features."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, TensorDataset

from urban_s2.pa_mcsp import (
    SafeResidualReranker,
    build_multicue_signals,
    candidate_errors_km,
    reranking_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--development-cache", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260724, 20260725, 20260726],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maximum-ram-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_ram_limit(limit: float = 0.70) -> None:
    used = psutil.virtual_memory().percent / 100.0
    if used >= float(limit):
        raise RuntimeError(
            f"RAM guard stopped PA-MCSP at {used:.1%}; limit is {limit:.1%}."
        )


def candidate_arrays(
    cache: Any,
    *,
    gallery_coordinate_key: str,
    gallery_label_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prototype_indices = cache["prototype_indices"]
    retrieval_indices = cache["retrieval_indices"]
    leaf_counts = cache["leaf_counts"]
    gallery_labels = cache[gallery_label_key]
    prototype_latlon = cache["prototype_centers_latlon"][prototype_indices]
    retrieval_latlon = cache[gallery_coordinate_key][retrieval_indices]
    prototype_density = leaf_counts[prototype_indices]
    retrieval_density = leaf_counts[gallery_labels[retrieval_indices]]
    signals, anchor_logits = build_multicue_signals(
        cache["prototype_scores"],
        cache["retrieval_scores"],
        prototype_latlon,
        retrieval_latlon,
        prototype_density,
        retrieval_density,
    )
    coordinates = np.concatenate((prototype_latlon, retrieval_latlon), axis=1)
    return signals, anchor_logits, coordinates, prototype_indices


def slice_union(values: np.ndarray, k: int, maximum_k: int) -> np.ndarray:
    return np.concatenate(
        (values[:, :k], values[:, maximum_k : maximum_k + k]),
        axis=1,
    )


def point_metrics(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "queries": int(len(values)),
        "accuracy_1km": float(np.mean(values <= 1.0)),
        "accuracy_25km": float(np.mean(values <= 25.0)),
        "accuracy_200km": float(np.mean(values <= 200.0)),
        "accuracy_750km": float(np.mean(values <= 750.0)),
        "accuracy_2500km": float(np.mean(values <= 2500.0)),
        "median_error_km": float(np.median(values)),
        "mean_error_km": float(np.mean(values)),
        "p75_error_km": float(np.quantile(values, 0.75)),
        "p90_error_km": float(np.quantile(values, 0.90)),
        "geoscore": float(np.mean(np.exp(-values / 1492.7))),
        "mean_log1p_error": float(np.mean(np.log1p(values))),
    }


def predict_scores(
    model: SafeResidualReranker,
    signals: np.ndarray,
    anchor: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            standardized = (
                signals[start : start + batch_size] - mean
            ) / scale
            logits, residual = model(
                torch.from_numpy(standardized.astype(np.float32)),
                torch.from_numpy(
                    anchor[start : start + batch_size].astype(np.float32)
                ),
            )
            predictions.append(logits.argmax(dim=1).numpy())
            residual_values.append(residual.numpy())
    return np.concatenate(predictions), np.concatenate(residual_values)


def errors_from_choice(
    candidate_errors: np.ndarray,
    choices: np.ndarray,
) -> np.ndarray:
    return candidate_errors[np.arange(len(candidate_errors)), choices]


def cluster_bootstrap_difference(
    proposed: np.ndarray,
    baseline: np.ndarray,
    clusters: np.ndarray,
    *,
    transform: str,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if transform == "accuracy_25km":
        differences = (
            (np.asarray(proposed) <= 25.0).astype(np.float64)
            - (np.asarray(baseline) <= 25.0).astype(np.float64)
        )
    elif transform == "log1p_error":
        differences = np.log1p(proposed) - np.log1p(baseline)
    else:
        raise ValueError(f"Unknown bootstrap transform: {transform}")
    unique, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    sizes = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, differences)
    np.add.at(sizes, inverse, 1.0)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        selected = generator.integers(0, len(unique), size=len(unique))
        draws[draw] = sums[selected].sum() / sizes[selected].sum()
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def train_one(
    *,
    signals: np.ndarray,
    anchor: np.ndarray,
    errors: np.ndarray,
    roles: np.ndarray,
    k: int,
    maximum_k: int,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[SafeResidualReranker, np.ndarray, np.ndarray, dict[str, Any]]:
    set_seed(seed)
    selected_signals = slice_union(signals, k, maximum_k)
    selected_anchor = slice_union(anchor, k, maximum_k)
    selected_errors = slice_union(errors, k, maximum_k)
    train_mask = roles == "reranker_train"
    calibration_mask = roles == "calibration"
    mean = selected_signals[train_mask].mean(axis=(0, 1), keepdims=True)
    scale = selected_signals[train_mask].std(axis=(0, 1), keepdims=True)
    scale = np.maximum(scale, 1e-5)
    standardized_train = (
        selected_signals[train_mask] - mean
    ) / scale
    dataset = TensorDataset(
        torch.from_numpy(standardized_train.astype(np.float32)),
        torch.from_numpy(selected_anchor[train_mask].astype(np.float32)),
        torch.from_numpy(selected_errors[train_mask].astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
    )
    model = SafeResidualReranker(
        signal_dimension=selected_signals.shape[-1],
        hidden_dimension=32,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_objective = float("inf")
    stale_epochs = 0
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        loss_sum = 0.0
        rows = 0
        for batch_signals, batch_anchor, batch_errors in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(batch_signals, batch_anchor)
            loss = reranking_loss(
                logits,
                residual,
                batch_errors,
                expected_error_weight=0.2,
                anchor_weight=0.05,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("PA-MCSP produced a non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_signals)
            rows += len(batch_signals)

        calibration_choices, calibration_residual = predict_scores(
            model,
            selected_signals[calibration_mask],
            selected_anchor[calibration_mask],
            mean,
            scale,
            batch_size=batch_size,
        )
        calibration_errors = errors_from_choice(
            selected_errors[calibration_mask],
            calibration_choices,
        )
        metrics = point_metrics(calibration_errors)
        objective = float(metrics["mean_log1p_error"])
        records.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / rows,
                "calibration": metrics,
                "mean_abs_residual": float(
                    np.mean(np.abs(calibration_residual))
                ),
            }
        )
        if objective < best_objective - 1e-5:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        enforce_ram_limit()
        if stale_epochs >= int(patience):
            break
    model.load_state_dict(best_state)
    final_choices, final_residual = predict_scores(
        model,
        selected_signals[calibration_mask],
        selected_anchor[calibration_mask],
        mean,
        scale,
        batch_size=batch_size,
    )
    final_errors = errors_from_choice(
        selected_errors[calibration_mask],
        final_choices,
    )
    profile = {
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(records),
        "best_mean_log1p_error": best_objective,
        "calibration": point_metrics(final_errors),
        "mean_abs_residual": float(np.mean(np.abs(final_residual))),
        "epoch_records": records,
    }
    return model, mean, scale, profile


def main() -> None:
    args = parse_args()
    threads = max(1, min(int(args.threads), os.cpu_count() or 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    enforce_ram_limit(args.maximum_ram_fraction)
    started = time.perf_counter()
    # The cache is generated and hash-locked by our own preparation stage.
    # ``allow_pickle`` preserves compatibility with the first cache revision,
    # which serialized string roles as an object array.
    training_cache = np.load(args.training_cache, allow_pickle=True)
    development_cache = np.load(args.development_cache, allow_pickle=False)
    feature_index = pd.read_parquet(args.feature_index)
    development_index = feature_index.loc[
        feature_index["split"].astype(str).eq("development")
    ].reset_index(drop=True)
    if bool(feature_index["official_split"].astype(str).eq("test").any()):
        raise RuntimeError("Official test rows are forbidden.")
    if not np.array_equal(
        development_index["feature_row"].to_numpy(dtype=np.int64),
        development_cache["development_feature_rows"],
    ):
        raise RuntimeError("Development cache row order differs from index.")

    training_signals, training_anchor, training_coordinates, _ = (
        candidate_arrays(
            training_cache,
            gallery_coordinate_key="gallery_coordinates_latlon",
            gallery_label_key="gallery_partition_labels",
        )
    )
    training_errors = candidate_errors_km(
        training_cache["query_latlon"],
        training_coordinates,
    ).astype(np.float32)
    enforce_ram_limit(args.maximum_ram_fraction)
    development_signals, development_anchor, development_coordinates, _ = (
        candidate_arrays(
            development_cache,
            gallery_coordinate_key="train_coordinates_latlon",
            gallery_label_key="train_partition_labels",
        )
    )
    development_truth = development_index[
        ["latitude", "longitude"]
    ].to_numpy()
    development_errors = candidate_errors_km(
        development_truth,
        development_coordinates,
    ).astype(np.float32)
    enforce_ram_limit(args.maximum_ram_fraction)

    maximum_training_k = training_cache["prototype_indices"].shape[1]
    maximum_development_k = development_cache["prototype_indices"].shape[1]
    top_k_values = sorted({int(value) for value in args.top_k})
    if max(top_k_values) > min(maximum_training_k, maximum_development_k):
        raise ValueError("Requested K exceeds a candidate cache.")
    roles = training_cache["query_roles"].astype(str)
    sequences = development_index["sequence"].astype(str).to_numpy()
    prototype_baseline = development_errors[:, 0]
    retrieval_baseline = development_errors[:, maximum_development_k]
    raw_anchor_choice = development_anchor.argmax(axis=1)
    raw_anchor_errors = errors_from_choice(
        development_errors,
        raw_anchor_choice,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    all_results: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    models: dict[tuple[int, int], tuple[Any, np.ndarray, np.ndarray]] = {}
    for k in top_k_values:
        k_results: dict[str, Any] = {}
        for seed in args.seeds:
            model, mean, scale, training_profile = train_one(
                signals=training_signals,
                anchor=training_anchor,
                errors=training_errors,
                roles=roles,
                k=k,
                maximum_k=maximum_training_k,
                seed=int(seed),
                epochs=int(args.epochs),
                patience=int(args.patience),
                batch_size=int(args.batch_size),
            )
            dev_signals_k = slice_union(
                development_signals,
                k,
                maximum_development_k,
            )
            dev_anchor_k = slice_union(
                development_anchor,
                k,
                maximum_development_k,
            )
            dev_errors_k = slice_union(
                development_errors,
                k,
                maximum_development_k,
            )
            choices, residual = predict_scores(
                model,
                dev_signals_k,
                dev_anchor_k,
                mean,
                scale,
                batch_size=int(args.batch_size),
            )
            proposed_errors = errors_from_choice(dev_errors_k, choices)
            accuracy_bootstrap = cluster_bootstrap_difference(
                proposed_errors,
                retrieval_baseline,
                sequences,
                transform="accuracy_25km",
                resamples=int(args.bootstrap_resamples),
                seed=int(seed) + 101,
            )
            log_error_bootstrap = cluster_bootstrap_difference(
                proposed_errors,
                prototype_baseline,
                sequences,
                transform="log1p_error",
                resamples=int(args.bootstrap_resamples),
                seed=int(seed) + 211,
            )
            metrics = point_metrics(proposed_errors)
            metrics["training"] = training_profile
            metrics["mean_abs_residual"] = float(np.mean(np.abs(residual)))
            metrics["bootstrap_accuracy_25km_vs_retrieval"] = (
                accuracy_bootstrap
            )
            metrics["bootstrap_log1p_error_vs_prototype"] = (
                log_error_bootstrap
            )
            k_results[str(seed)] = metrics
            checkpoint_path = checkpoint_root / f"k{k}_seed_{seed}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "method": "pa_mcsp_safe_residual_reranker",
                    "k_per_source": k,
                    "seed": int(seed),
                    "signal_mean": mean,
                    "signal_scale": scale,
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            frame = development_index[
                ["id", "sequence", "latitude", "longitude", "country"]
            ].copy()
            frame["k_per_source"] = k
            frame["seed"] = int(seed)
            frame["error_km"] = proposed_errors
            frame["chosen_candidate"] = choices
            prediction_frames.append(frame)
            models[(k, int(seed))] = (model, mean, scale)
        all_results[str(k)] = k_results
        enforce_ram_limit(args.maximum_ram_fraction)

    calibration_objective = {
        str(k): float(
            np.mean(
                [
                    result["training"]["best_mean_log1p_error"]
                    for result in all_results[str(k)].values()
                ]
            )
        )
        for k in top_k_values
    }
    selected_k = int(min(calibration_objective, key=calibration_objective.get))
    selected_results = all_results[str(selected_k)]
    seed_passes: dict[str, bool] = {}
    for seed, result in selected_results.items():
        seed_passes[seed] = bool(
            result["accuracy_25km"] >= 0.24045
            and result["median_error_km"] <= 186.85
            and result["geoscore"] >= 0.77729
            and result["bootstrap_accuracy_25km_vs_retrieval"]["ci95_low"] > 0
            and result["bootstrap_log1p_error_vs_prototype"]["ci95_high"] < 0
        )
    payload = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "method": "pa_mcsp_safe_residual_multicue_reranker",
        "official_test_evaluated": False,
        "development_used_for_model_selection": False,
        "counts": {
            "reranker_train": int(np.sum(roles == "reranker_train")),
            "calibration": int(np.sum(roles == "calibration")),
            "development": int(len(development_index)),
        },
        "baselines": {
            "prototype": point_metrics(prototype_baseline),
            "retrieval": point_metrics(retrieval_baseline),
            "raw_union_anchor": point_metrics(raw_anchor_errors),
        },
        "selection": {
            "criterion": "mean_seed_calibration_mean_log1p_error",
            "calibration_objective": calibration_objective,
            "selected_k_per_source": selected_k,
            "seeds": [int(seed) for seed in args.seeds],
        },
        "gate": {
            "thresholds": {
                "accuracy_25km_minimum": 0.24045,
                "median_error_km_maximum": 186.85,
                "geoscore_minimum": 0.77729,
                "accuracy_ci95_low_vs_retrieval_minimum_exclusive": 0.0,
                "log1p_error_ci95_high_vs_prototype_maximum_exclusive": 0.0,
            },
            "seed_passes": seed_passes,
            "passed": bool(seed_passes and all(seed_passes.values())),
            "decision": (
                "ALLOW_25PCT"
                if seed_passes and all(seed_passes.values())
                else "REJECT_SCALE_UP"
            ),
        },
        "results": all_results,
        "profiling": {
            "total_seconds": time.perf_counter() - started,
            "threads": threads,
            "ram_fraction_end": psutil.virtual_memory().percent / 100.0,
            "maximum_ram_fraction": float(args.maximum_ram_fraction),
        },
        "provenance": {
            "training_cache": str(args.training_cache),
            "training_cache_sha256": sha256_file(args.training_cache),
            "development_cache": str(args.development_cache),
            "development_cache_sha256": sha256_file(args.development_cache),
            "feature_index": str(args.feature_index),
            "feature_index_sha256": sha256_file(args.feature_index),
        },
    }
    metrics_path = args.output_root / "pa_mcsp_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(
        args.output_root / "pa_mcsp_development_predictions.parquet",
        index=False,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
