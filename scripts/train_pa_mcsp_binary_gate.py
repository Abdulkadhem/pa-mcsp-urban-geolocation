#!/usr/bin/env python3
"""Train a safe binary PA-MCSP gate between top-1 source hypotheses."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from urban_s2.pa_mcsp import (
    SafeBinaryGate,
    build_multicue_signals,
    candidate_errors_km,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--development-cache", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260724, 20260725, 20260726],
    )
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Training/inference device; default preserves CPU-only behavior.",
    )
    parser.add_argument("--normalization", choices=["standard", "robust"], default="standard")
    parser.add_argument("--country-balanced", action="store_true")
    parser.add_argument("--maximum-ram-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--allow-decision-label",
        default="ALLOW_25PCT",
        help="Decision label emitted when every seed passes the scale gate.",
    )
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


def point_metrics(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "queries": int(len(values)),
        "accuracy_1km": float(np.mean(values <= 1.0)),
        "accuracy_25km": float(np.mean(values <= 25.0)),
        "accuracy_200km": float(np.mean(values <= 200.0)),
        "median_error_km": float(np.median(values)),
        "mean_error_km": float(np.mean(values)),
        "p75_error_km": float(np.quantile(values, 0.75)),
        "p90_error_km": float(np.quantile(values, 0.90)),
        "geoscore": float(np.mean(np.exp(-values / 1492.7))),
        "mean_log1p_error": float(np.mean(np.log1p(values))),
    }


def cache_top1_data(
    cache: Any,
    *,
    coordinate_key: str,
    label_key: str,
    support_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maximum_k = cache["prototype_indices"].shape[1]
    k = min(int(support_k), maximum_k)
    prototype_indices = cache["prototype_indices"][:, :k]
    retrieval_indices = cache["retrieval_indices"][:, :k]
    leaf_counts = cache["leaf_counts"]
    gallery_labels = cache[label_key]
    prototype_points = cache["prototype_centers_latlon"][prototype_indices]
    retrieval_points = cache[coordinate_key][retrieval_indices]
    prototype_density = leaf_counts[prototype_indices]
    retrieval_density = leaf_counts[gallery_labels[retrieval_indices]]
    signals, _ = build_multicue_signals(
        cache["prototype_scores"][:, :k],
        cache["retrieval_scores"][:, :k],
        prototype_points,
        retrieval_points,
        prototype_density,
        retrieval_density,
    )
    top_prototype = signals[:, 0]
    top_retrieval = signals[:, k]
    query_signals = np.concatenate(
        (
            top_prototype,
            top_retrieval,
            top_retrieval - top_prototype,
        ),
        axis=1,
    ).astype(np.float32)
    anchor_log_odds = (
        (
            cache["retrieval_scores"][:, 0]
            - cache["prototype_scores"][:, 0]
        )
        / 0.02
    ).astype(np.float32)
    truth_key = (
        "query_latlon"
        if "query_latlon" in cache.files
        else "development_truth_unused"
    )
    return (
        query_signals,
        anchor_log_odds,
        prototype_points[:, 0],
        retrieval_points[:, 0],
    )


def choose_errors(
    gate_logits: np.ndarray,
    prototype_errors: np.ndarray,
    retrieval_errors: np.ndarray,
) -> np.ndarray:
    return np.where(gate_logits > 0.0, retrieval_errors, prototype_errors)


def predict_gate(
    model: SafeBinaryGate,
    signals: np.ndarray,
    anchor: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            standardized = (
                signals[start : start + batch_size] - mean
            ) / scale
            batch_logits, batch_residual = model(
                torch.from_numpy(standardized.astype(np.float32)).to(device),
                torch.from_numpy(anchor[start : start + batch_size].astype(np.float32)).to(device),
            )
            logits.append(batch_logits.detach().cpu().numpy())
            residuals.append(batch_residual.detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(residuals)


def bootstrap_difference(
    proposed: np.ndarray,
    baseline: np.ndarray,
    clusters: np.ndarray,
    *,
    accuracy: bool,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    differences = (
        (proposed <= 25.0).astype(float) - (baseline <= 25.0).astype(float)
        if accuracy
        else np.log1p(proposed) - np.log1p(baseline)
    )
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    sizes = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, differences)
    np.add.at(sizes, inverse, 1.0)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        selected = generator.integers(0, len(unique), size=len(unique))
        draws[index] = sums[selected].sum() / sizes[selected].sum()
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def train_seed(
    *,
    signals: np.ndarray,
    anchor: np.ndarray,
    prototype_errors: np.ndarray,
    retrieval_errors: np.ndarray,
    roles: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    groups: np.ndarray,
    normalization: str,
    country_balanced: bool,
) -> tuple[SafeBinaryGate, np.ndarray, np.ndarray, dict[str, Any]]:
    set_seed(seed)
    train_mask = roles == "reranker_train"
    calibration_mask = roles == "calibration"
    if normalization == "robust":
        mean = np.median(signals[train_mask], axis=0, keepdims=True)
        scale = np.maximum(np.percentile(signals[train_mask], 75, axis=0, keepdims=True) - np.percentile(signals[train_mask], 25, axis=0, keepdims=True), 1e-5)
    else:
        mean = signals[train_mask].mean(axis=0, keepdims=True)
        scale = np.maximum(signals[train_mask].std(axis=0, keepdims=True), 1e-5)
    standardized = (signals[train_mask] - mean) / scale
    target = (
        retrieval_errors[train_mask] < prototype_errors[train_mask]
    ).astype(np.float32)
    importance = np.clip(
        np.abs(
            np.log1p(retrieval_errors[train_mask])
            - np.log1p(prototype_errors[train_mask])
        ),
        0.1,
        4.0,
    ).astype(np.float32)
    if country_balanced:
        group_values, group_counts = np.unique(groups[train_mask], return_counts=True)
        weights = {str(g): float(len(groups[train_mask]) / max(1, len(group_values) * c)) ** 0.5 for g, c in zip(group_values, group_counts)}
        group_weight = np.asarray([weights.get(str(g), 1.0) for g in groups[train_mask]], dtype=np.float32)
        group_weight = np.clip(group_weight, 0.5, 3.0)
        importance *= group_weight
    dataset = TensorDataset(
        torch.from_numpy(standardized.astype(np.float32)),
        torch.from_numpy(anchor[train_mask].astype(np.float32)),
        torch.from_numpy(target),
        torch.from_numpy(importance),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = SafeBinaryGate(signal_dimension=signals.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline_errors = choose_errors(
        anchor[calibration_mask],
        prototype_errors[calibration_mask],
        retrieval_errors[calibration_mask],
    )
    baseline_metrics = point_metrics(baseline_errors)
    best_objective = float(baseline_metrics["mean_log1p_error"])
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = baseline_metrics
    selected_from_training = False
    stale = 0
    records: list[dict[str, Any]] = [
        {"epoch": -1, "calibration": baseline_metrics, "safe_anchor": True}
    ]
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        rows = 0
        for batch_signals, batch_anchor, batch_target, batch_importance in loader:
            batch_signals = batch_signals.to(device, non_blocking=True)
            batch_anchor = batch_anchor.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_importance = batch_importance.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(batch_signals, batch_anchor)
            classification = F.binary_cross_entropy_with_logits(
                logits,
                batch_target,
                weight=batch_importance,
            )
            loss = classification + 0.05 * residual.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_signals)
            rows += len(batch_signals)
        calibration_logits, calibration_residual = predict_gate(
            model,
            signals[calibration_mask],
            anchor[calibration_mask],
            mean,
            scale,
            batch_size,
            device,
        )
        calibration_errors = choose_errors(
            calibration_logits,
            prototype_errors[calibration_mask],
            retrieval_errors[calibration_mask],
        )
        metrics = point_metrics(calibration_errors)
        safe = bool(
            metrics["accuracy_25km"] >= baseline_metrics["accuracy_25km"]
            and metrics["median_error_km"] <= baseline_metrics["median_error_km"]
            and metrics["geoscore"] >= baseline_metrics["geoscore"]
        )
        records.append(
            {
                "epoch": epoch,
                "training_loss": total / rows,
                "calibration": metrics,
                "safe_vs_anchor": safe,
                "mean_abs_residual": float(
                    np.mean(np.abs(calibration_residual))
                ),
            }
        )
        objective = float(metrics["mean_log1p_error"])
        if safe and objective < best_objective - 1e-5:
            best_objective = objective
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            selected_from_training = True
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, mean, scale, {
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(records) - 1,
        "anchor_calibration": baseline_metrics,
        "selected_calibration": best_metrics,
        "selected_anchor_fallback": not selected_from_training,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    threads = max(1, min(args.threads, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    started = time.perf_counter()
    training = np.load(args.training_cache, allow_pickle=True)
    development = np.load(args.development_cache, allow_pickle=False)
    index = pd.read_parquet(args.feature_index)
    query_rows = training["query_feature_rows"].astype(np.int64)
    groups = index.iloc[query_rows]["country"].fillna("UNKNOWN").astype(str).to_numpy()
    development_index = index.loc[
        index["split"].astype(str).eq("development")
    ].reset_index(drop=True)
    if bool(index["official_split"].astype(str).eq("test").any()):
        raise RuntimeError("Official test rows are forbidden.")

    train_signals, train_anchor, train_prototype, train_retrieval = (
        cache_top1_data(
            training,
            coordinate_key="gallery_coordinates_latlon",
            label_key="gallery_partition_labels",
            support_k=args.support_k,
        )
    )
    train_truth = training["query_latlon"]
    train_prototype_errors = candidate_errors_km(
        train_truth,
        train_prototype[:, None],
    )[:, 0]
    train_retrieval_errors = candidate_errors_km(
        train_truth,
        train_retrieval[:, None],
    )[:, 0]
    dev_signals, dev_anchor, dev_prototype, dev_retrieval = cache_top1_data(
        development,
        coordinate_key="train_coordinates_latlon",
        label_key="train_partition_labels",
        support_k=args.support_k,
    )
    dev_truth = development_index[["latitude", "longitude"]].to_numpy()
    dev_prototype_errors = candidate_errors_km(
        dev_truth,
        dev_prototype[:, None],
    )[:, 0]
    dev_retrieval_errors = candidate_errors_km(
        dev_truth,
        dev_retrieval[:, None],
    )[:, 0]
    raw_errors = choose_errors(
        dev_anchor,
        dev_prototype_errors,
        dev_retrieval_errors,
    )
    roles = training["query_roles"].astype(str)
    sequences = development_index["sequence"].astype(str).to_numpy()
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        model, mean, scale, profile = train_seed(
            signals=train_signals,
            anchor=train_anchor,
            prototype_errors=train_prototype_errors,
            retrieval_errors=train_retrieval_errors,
            roles=roles,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=device,
            groups=groups,
            normalization=args.normalization,
            country_balanced=args.country_balanced,
        )
        logits, residual = predict_gate(
            model,
            dev_signals,
            dev_anchor,
            mean,
            scale,
            args.batch_size,
            device,
        )
        errors = choose_errors(
            logits,
            dev_prototype_errors,
            dev_retrieval_errors,
        )
        metrics = point_metrics(errors)
        metrics["training"] = profile
        metrics["mean_abs_residual"] = float(np.mean(np.abs(residual)))
        metrics["bootstrap_accuracy_25km_vs_retrieval"] = (
            bootstrap_difference(
                errors,
                dev_retrieval_errors,
                sequences,
                accuracy=True,
                resamples=args.bootstrap_resamples,
                seed=seed + 101,
            )
        )
        metrics["bootstrap_log1p_error_vs_prototype"] = bootstrap_difference(
            errors,
            dev_prototype_errors,
            sequences,
            accuracy=False,
            resamples=args.bootstrap_resamples,
            seed=seed + 211,
        )
        results[str(seed)] = metrics
        torch.save(
            {
                "schema_version": 1,
                "method": "pa_mcsp_safe_binary_gate",
                "seed": seed,
                "signal_mean": mean,
                "signal_scale": scale,
                "model_state": model.state_dict(),
                "metrics": metrics,
            },
            checkpoints / f"binary_gate_seed_{seed}.pt",
        )
        frame = development_index[
            ["id", "sequence", "latitude", "longitude", "country"]
        ].copy()
        frame["seed"] = seed
        frame["error_km"] = errors
        frame["selected_retrieval"] = logits > 0.0
        prediction_frames.append(frame)

    seed_passes = {
        seed: bool(
            result["accuracy_25km"] >= 0.24045
            and result["median_error_km"] <= 186.85
            and result["geoscore"] >= 0.77729
            and result["bootstrap_accuracy_25km_vs_retrieval"]["ci95_low"] > 0
            and result["bootstrap_log1p_error_vs_prototype"]["ci95_high"] < 0
        )
        for seed, result in results.items()
    }
    payload = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "method": "pa_mcsp_safe_binary_source_gate",
        "official_test_evaluated": False,
        "training_settings": {"normalization": args.normalization, "country_balanced": bool(args.country_balanced), "device": str(device)},
        "development_used_for_model_selection": False,
        "baselines": {
            "prototype": point_metrics(dev_prototype_errors),
            "retrieval": point_metrics(dev_retrieval_errors),
            "raw_union_anchor": point_metrics(raw_errors),
        },
        "results": results,
        "gate": {
            "seed_passes": seed_passes,
            "passed": bool(seed_passes and all(seed_passes.values())),
            "decision": (
                args.allow_decision_label
                if seed_passes and all(seed_passes.values())
                else "REJECT_SCALE_UP"
            ),
        },
        "profiling": {
            "seconds": time.perf_counter() - started,
            "threads": threads,
            "ram_fraction_end": psutil.virtual_memory().percent / 100.0,
        },
        "provenance": {
            "training_cache_sha256": sha256_file(args.training_cache),
            "development_cache_sha256": sha256_file(args.development_cache),
            "feature_index_sha256": sha256_file(args.feature_index),
        },
    }
    (args.output_root / "pa_mcsp_binary_gate_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        args.output_root / "pa_mcsp_binary_gate_predictions.parquet",
        index=False,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
