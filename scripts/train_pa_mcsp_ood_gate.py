"""Experimental OOD-aware PA-MCSP gate.

This is deliberately development-only.  It augments the frozen PA-MCSP
signals with confidence, agreement, dispersion and score-gap summaries, then
uses the same safe residual gate and pre-registered scale-up criteria as the
baseline.  The official test is never read by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from urban_s2.pa_mcsp import candidate_errors_km
from scripts.train_pa_mcsp_binary_gate import (
    bootstrap_difference,
    cache_top1_data,
    choose_errors,
    point_metrics,
    train_seed,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--training-cache", type=Path, required=True)
    p.add_argument("--development-cache", type=Path, required=True)
    p.add_argument("--feature-index", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--support-k", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[20260724, 20260725, 20260726])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--normalization", choices=["standard", "robust"], default="robust")
    p.add_argument("--country-balanced", action="store_true")
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    return p.parse_args()


def augment(signals: np.ndarray) -> np.ndarray:
    """Add inference-time OOD summaries without using coordinates/truth."""
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 3 != 0:
        raise ValueError(f"expected 3 equal PA-MCSP blocks, got {x.shape}")
    block = x.shape[1] // 3
    p, r, d = x[:, :block], x[:, block:2 * block], x[:, 2 * block:]
    # Per-source confidence and cross-source consistency channels.  The
    # original channels already contain score, rank, density and geographic
    # support; these summaries make their OOD interaction explicit.
    p_abs = np.abs(p)
    r_abs = np.abs(r)
    d_abs = np.abs(d)
    summaries = np.column_stack(
        [
            p[:, 0], p[:, 1], p[:, 2], p[:, 5], p[:, 6], p[:, 7], p[:, 8], p[:, 9], p[:, 10], p[:, 11],
            r[:, 0], r[:, 1], r[:, 2], r[:, 5], r[:, 6], r[:, 7], r[:, 8], r[:, 9], r[:, 10], r[:, 11],
            d[:, 0], d[:, 1], d[:, 2], d[:, 5], d[:, 6], d[:, 7], d[:, 8], d[:, 9], d[:, 10], d[:, 11],
            p_abs.mean(1), p_abs.std(1), r_abs.mean(1), r_abs.std(1),
            d_abs.mean(1), d_abs.std(1), d[:, 0] - d[:, 1],
            p[:, 9:12].mean(1), r[:, 9:12].mean(1), d[:, 9:12].mean(1),
        ]
    )
    return np.concatenate((x, summaries.astype(np.float32)), axis=1)


def main() -> None:
    a = args()
    torch.set_num_threads(max(1, min(a.threads, torch.get_num_threads())))
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train = np.load(a.training_cache, allow_pickle=True)
    dev = np.load(a.development_cache, allow_pickle=False)
    index = pd.read_parquet(a.feature_index)
    if bool(index["official_split"].astype(str).eq("test").any()):
        raise RuntimeError("official test rows are forbidden")
    roles = train["query_roles"].astype(str)
    query_rows = train["query_feature_rows"].astype(np.int64)
    groups = index.iloc[query_rows]["country"].fillna("UNKNOWN").astype(str).to_numpy()
    # Candidate cache is a deterministic development subset, not necessarily
    # every row carrying the development split label in the full index.
    dev_rows = dev["development_feature_rows"].astype(np.int64) if "development_feature_rows" in dev.files else np.flatnonzero(index["split"].astype(str).eq("development"))
    dev_index = index.iloc[dev_rows].reset_index(drop=True)
    ts = cache_top1_data(train, coordinate_key="gallery_coordinates_latlon", label_key="gallery_partition_labels", support_k=a.support_k)
    ds = cache_top1_data(dev, coordinate_key="train_coordinates_latlon", label_key="train_partition_labels", support_k=a.support_k)
    train_signals, train_anchor, train_p, train_r = ts
    dev_signals, dev_anchor, dev_p, dev_r = ds
    train_signals, dev_signals = augment(train_signals), augment(dev_signals)
    train_truth = train["query_latlon"]
    dev_truth = dev_index[["latitude", "longitude"]].to_numpy()
    train_pe = candidate_errors_km(train_truth, train_p[:, None])[:, 0]
    train_re = candidate_errors_km(train_truth, train_r[:, None])[:, 0]
    dev_pe = candidate_errors_km(dev_truth, dev_p[:, None])[:, 0]
    dev_re = candidate_errors_km(dev_truth, dev_r[:, None])[:, 0]
    a.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for seed in a.seeds:
        model, mean, scale, profile = train_seed(
            signals=train_signals, anchor=train_anchor,
            prototype_errors=train_pe, retrieval_errors=train_re,
            roles=roles, seed=seed, epochs=a.epochs, patience=a.patience,
            batch_size=a.batch_size, device=device, groups=groups,
            normalization=a.normalization, country_balanced=a.country_balanced,
        )
        # Reuse the prediction path with the augmented signal matrix.
        from scripts.train_pa_mcsp_binary_gate import predict_gate
        logits, residual = predict_gate(model, dev_signals, dev_anchor, mean, scale, a.batch_size, device)
        errors = choose_errors(logits, dev_pe, dev_re)
        metrics = point_metrics(errors)
        metrics["training"] = profile
        metrics["mean_abs_residual"] = float(np.abs(residual).mean())
        seq = dev_index["sequence"].astype(str).to_numpy()
        metrics["bootstrap_accuracy_25km_vs_retrieval"] = bootstrap_difference(errors, dev_re, seq, accuracy=True, resamples=a.bootstrap_resamples, seed=seed + 101)
        metrics["bootstrap_log1p_error_vs_prototype"] = bootstrap_difference(errors, dev_pe, seq, accuracy=False, resamples=a.bootstrap_resamples, seed=seed + 211)
        results[str(seed)] = metrics
        torch.save({"schema_version": 1, "method": "pa_mcsp_ood_gate", "seed": seed, "signal_mean": mean, "signal_scale": scale, "model_state": model.state_dict(), "metrics": metrics}, a.output_root / f"ood_gate_seed_{seed}.pt")
    passes = {s: bool(v["accuracy_25km"] >= 0.24045 and v["median_error_km"] <= 186.85 and v["geoscore"] >= 0.77729 and v["bootstrap_accuracy_25km_vs_retrieval"]["ci95_low"] > 0 and v["bootstrap_log1p_error_vs_prototype"]["ci95_high"] < 0) for s, v in results.items()}
    payload = {"schema_version": 1, "method": "pa_mcsp_ood_confidence_gate", "official_test_evaluated": False, "training_settings": {"support_k": a.support_k, "normalization": a.normalization, "country_balanced": bool(a.country_balanced), "device": str(device), "signal_dimension": int(train_signals.shape[1])}, "results": results, "gate": {"seed_passes": passes, "passed": bool(passes and all(passes.values())), "decision": "ALLOW_25PCT" if passes and all(passes.values()) else "REJECT_SCALE_UP"}, "baselines": {"prototype": point_metrics(dev_pe), "retrieval": point_metrics(dev_re)} }
    (a.output_root / "pa_mcsp_ood_gate_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
