#!/usr/bin/env python3
"""Evaluate frozen PA-MCSP on the locked official OSV5M test split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from urban_s2.cells import build_adaptive_s2_partition
from urban_s2.geodesy import unit_to_latlon
from urban_s2.pa_mcsp import SafeBinaryGate, build_multicue_signals, candidate_errors_km, topk_cosine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--train-index", type=Path, required=True)
    p.add_argument("--test-features", type=Path, required=True)
    p.add_argument("--test-index", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-threshold", type=float, default=0.0)
    p.add_argument("--gate-retrieval-rate", type=float, default=None, help="Optional unlabeled test-time quantile rate, calibrated from development.")
    p.add_argument("--ood-gate", action="store_true", help="Augment PA-MCSP gate inputs with OOD confidence summaries.")
    return p.parse_args()


def augment_ood(signals: np.ndarray) -> np.ndarray:
    """Match the development-only OOD gate feature construction exactly."""
    x = np.asarray(signals, dtype=np.float32)
    block = x.shape[1] // 3
    p, r, d = x[:, :block], x[:, block:2 * block], x[:, 2 * block:]
    summaries = np.column_stack([
        p[:, 0], p[:, 1], p[:, 2], p[:, 5], p[:, 6], p[:, 7], p[:, 8], p[:, 9], p[:, 10], p[:, 11],
        r[:, 0], r[:, 1], r[:, 2], r[:, 5], r[:, 6], r[:, 7], r[:, 8], r[:, 9], r[:, 10], r[:, 11],
        d[:, 0], d[:, 1], d[:, 2], d[:, 5], d[:, 6], d[:, 7], d[:, 8], d[:, 9], d[:, 10], d[:, 11],
        np.abs(p).mean(1), np.abs(p).std(1), np.abs(r).mean(1), np.abs(r).std(1),
        np.abs(d).mean(1), np.abs(d).std(1), d[:, 0] - d[:, 1],
        p[:, 9:12].mean(1), r[:, 9:12].mean(1), d[:, 9:12].mean(1),
    ])
    return np.concatenate((x, summaries.astype(np.float32)), axis=1)


def summary(errors: np.ndarray) -> dict[str, float | int]:
    e = np.asarray(errors, dtype=np.float64)
    return {"queries": int(len(e)), "accuracy_1km": float(np.mean(e <= 1)), "accuracy_25km": float(np.mean(e <= 25)), "accuracy_200km": float(np.mean(e <= 200)), "median_error_km": float(np.median(e)), "mean_error_km": float(np.mean(e)), "p75_error_km": float(np.quantile(e, .75)), "p90_error_km": float(np.quantile(e, .90)), "geoscore": float(np.mean(np.exp(-e / 1492.7))), "mean_log1p_error": float(np.mean(np.log1p(e)))}


def clustered_difference(proposed: np.ndarray, baseline: np.ndarray, groups: np.ndarray, *, log_error: bool, seed: int = 20260726) -> dict[str, float]:
    values = (np.log1p(proposed) - np.log1p(baseline)) if log_error else ((proposed <= 25).astype(float) - (baseline <= 25).astype(float))
    _, inverse = np.unique(groups.astype(str), return_inverse=True)
    n_groups = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=values, minlength=n_groups)
    counts = np.bincount(inverse, minlength=n_groups).astype(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(1000, dtype=float)
    for i in range(len(draws)):
        selected = rng.integers(0, n_groups, size=n_groups)
        draws[i] = sums[selected].sum() / counts[selected].sum()
    return {"difference": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975))}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    train_idx_all = pd.read_parquet(args.train_index)
    test_idx = pd.read_parquet(args.test_index)
    train_mask = train_idx_all["split"].astype(str).eq("train").to_numpy()
    train_idx = train_idx_all.loc[train_mask].reset_index(drop=True)
    train_features = np.asarray(np.load(args.train_features, mmap_mode="r")[train_mask], dtype=np.float32)
    test_features = np.asarray(np.load(args.test_features, mmap_mode="r"), dtype=np.float32)
    if len(test_features) != len(test_idx):
        raise RuntimeError("test feature/index mismatch")
    partition = build_adaptive_s2_partition(train_idx.latitude.to_numpy(), train_idx.longitude.to_numpy(), min_level=3, max_level=12, max_samples_per_leaf=64)
    sums = np.zeros((len(partition.leaf_counts), train_features.shape[1]), dtype=np.float64)
    np.add.at(sums, partition.labels, train_features.astype(np.float64))
    prototypes = (sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(test_features).to(device)
    proto_pool = topk_cosine(q, torch.from_numpy(prototypes).to(device), top_k=args.top_k, batch_size=args.batch_size)
    retrieval_pool = topk_cosine(q, torch.from_numpy(train_features).to(device), top_k=args.top_k, batch_size=args.batch_size)
    centers = unit_to_latlon(partition.leaf_centers_unit)
    train_coords = train_idx[["latitude", "longitude"]].to_numpy()
    truth = test_idx[["latitude", "longitude"]].to_numpy()
    proto_indices = np.asarray(proto_pool.indices)
    retrieval_indices = np.asarray(retrieval_pool.indices)
    proto_scores = np.asarray(proto_pool.scores)
    retrieval_scores = np.asarray(retrieval_pool.scores)
    proto_points = centers[proto_indices]
    retrieval_points = train_coords[retrieval_indices]
    leaf_counts = partition.leaf_counts
    proto_density = leaf_counts[proto_pool.indices]
    retrieval_density = leaf_counts[partition.labels[retrieval_indices]]
    signals, _ = build_multicue_signals(proto_scores, retrieval_scores, proto_points, retrieval_points, proto_density, retrieval_density)
    p_errors = candidate_errors_km(truth, proto_points[:, 0:1])[:, 0]
    r_errors = candidate_errors_km(truth, retrieval_points[:, 0:1])[:, 0]
    top_proto = signals[:, 0]
    top_ret = signals[:, args.top_k]
    gate_input = np.concatenate((top_proto, top_ret, top_ret - top_proto), axis=1).astype(np.float32)
    if args.ood_gate:
        gate_input = augment_ood(gate_input)
    anchor = ((proto_scores[:, 0] - retrieval_scores[:, 0]) / 0.02).astype(np.float32)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SafeBinaryGate(signal_dimension=gate_input.shape[1]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    mean = np.asarray(ckpt["signal_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["signal_scale"], dtype=np.float32)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(gate_input), args.batch_size):
            x = torch.from_numpy(((gate_input[start:start+args.batch_size] - mean) / scale).astype(np.float32)).to(device)
            a = torch.from_numpy(anchor[start:start+args.batch_size]).to(device)
            y, _ = model(x, a)
            logits.append(y.cpu().numpy())
    gate_logits = np.concatenate(logits)
    effective_threshold = float(args.gate_threshold)
    if args.gate_retrieval_rate is not None:
        if not 0.0 <= args.gate_retrieval_rate <= 1.0:
            raise ValueError("gate retrieval rate must be in [0,1]")
        effective_threshold = float(np.quantile(gate_logits, 1.0 - args.gate_retrieval_rate))
    gate_errors = np.where(gate_logits > effective_threshold, r_errors, p_errors)
    groups = test_idx["sequence"].astype(str).to_numpy() if "sequence" in test_idx else np.arange(len(test_idx)).astype(str)
    payload = {"schema_version": 1, "official_test_evaluated": True, "completed_utc": pd.Timestamp.utcnow().isoformat(), "device": str(device), "counts": {"train_gallery": int(len(train_idx)), "test_queries": int(len(test_idx)), "s2_leaves": int(len(partition.leaf_counts))}, "baselines": {"prototype": summary(p_errors), "retrieval": summary(r_errors)}, "pa_mcsp": summary(gate_errors), "paired_vs_prototype": {"accuracy_25km": clustered_difference(gate_errors, p_errors, groups, log_error=False), "log1p_error": clustered_difference(gate_errors, p_errors, groups, log_error=True)}, "paired_vs_retrieval": {"accuracy_25km": clustered_difference(gate_errors, r_errors, groups, log_error=False), "log1p_error": clustered_difference(gate_errors, r_errors, groups, log_error=True)}, "gate_threshold": effective_threshold, "gate_retrieval_rate_calibration": args.gate_retrieval_rate, "gate_retrieval_rate": float(np.mean(gate_logits > effective_threshold)), "checkpoint": str(args.checkpoint), "elapsed_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
