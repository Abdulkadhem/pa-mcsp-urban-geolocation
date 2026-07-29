#!/usr/bin/env python3
"""Evaluate cross-fitted SV-DS-PA-MCSP utility checkpoints on official test."""
from __future__ import annotations

import argparse
import glob
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
from urban_s2.pa_mcsp import UtilityAbstentionGate, build_multicue_signals, candidate_errors_km, topk_cosine
from scripts.train_pa_mcsp_ood_gate import augment


def summary(errors: np.ndarray) -> dict[str, float | int]:
    e = np.asarray(errors, dtype=np.float64)
    return {"queries": int(len(e)), "accuracy_1km": float(np.mean(e <= 1)), "accuracy_25km": float(np.mean(e <= 25)), "accuracy_200km": float(np.mean(e <= 200)), "median_error_km": float(np.median(e)), "mean_error_km": float(np.mean(e)), "p75_error_km": float(np.quantile(e, .75)), "p90_error_km": float(np.quantile(e, .90)), "geoscore": float(np.mean(np.exp(-e / 1492.7))), "mean_log1p_error": float(np.mean(np.log1p(e)))}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--train-index", type=Path, required=True)
    p.add_argument("--test-features", type=Path, required=True)
    p.add_argument("--test-index", type=Path, required=True)
    p.add_argument("--checkpoint-glob", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ensemble", action="store_true", help="Average the three seed models within each outer fold.")
    return p.parse_args()


def main() -> None:
    a = args()
    started = time.perf_counter()
    train_all = pd.read_parquet(a.train_index)
    test_idx = pd.read_parquet(a.test_index)
    train_mask = train_all["split"].astype(str).eq("train").to_numpy()
    train_idx = train_all.loc[train_mask].reset_index(drop=True)
    train_features = np.asarray(np.load(a.train_features, mmap_mode="r")[train_mask], dtype=np.float32)
    test_features = np.asarray(np.load(a.test_features, mmap_mode="r"), dtype=np.float32)
    if len(test_features) != len(test_idx):
        raise RuntimeError("test feature/index mismatch")
    partition = build_adaptive_s2_partition(train_idx.latitude.to_numpy(), train_idx.longitude.to_numpy(), min_level=3, max_level=12, max_samples_per_leaf=64)
    sums = np.zeros((len(partition.leaf_counts), train_features.shape[1]), dtype=np.float64)
    np.add.at(sums, partition.labels, train_features.astype(np.float64))
    prototypes = (sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(test_features).to(device)
    proto_pool = topk_cosine(q, torch.from_numpy(prototypes).to(device), top_k=a.top_k, batch_size=a.batch_size)
    retrieval_pool = topk_cosine(q, torch.from_numpy(train_features).to(device), top_k=a.top_k, batch_size=a.batch_size)
    centers = unit_to_latlon(partition.leaf_centers_unit)
    train_coords = train_idx[["latitude", "longitude"]].to_numpy()
    truth = test_idx[["latitude", "longitude"]].to_numpy()
    proto_points, retrieval_points = centers[proto_pool.indices], train_coords[retrieval_pool.indices]
    proto_scores, retrieval_scores = np.asarray(proto_pool.scores), np.asarray(retrieval_pool.scores)
    proto_density = partition.leaf_counts[proto_pool.indices]
    retrieval_density = partition.leaf_counts[partition.labels[retrieval_pool.indices]]
    signals, _ = build_multicue_signals(proto_scores, retrieval_scores, proto_points, retrieval_points, proto_density, retrieval_density)
    # Match cache_top1_data: retain top prototype and top retrieval cue blocks.
    top_proto, top_retrieval = signals[:, 0], signals[:, a.top_k]
    signals = augment(np.concatenate((top_proto, top_retrieval, top_retrieval - top_proto), axis=1).astype(np.float32))
    anchor = ((proto_scores[:, 0] - retrieval_scores[:, 0]) / 0.02).astype(np.float32)
    p_errors = candidate_errors_km(truth, proto_points[:, :1])[:, 0]
    r_errors = candidate_errors_km(truth, retrieval_points[:, :1])[:, 0]
    paths = [Path(p) for p in sorted(glob.glob(a.checkpoint_glob))]
    groups = [[p] for p in paths]
    if a.ensemble:
        by_fold = {}
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            by_fold.setdefault(int(ck["outer_fold"]), []).append(p)
        groups = [by_fold[k] for k in sorted(by_fold)]
    rows = []
    for group in groups:
        logits_all, delta_all = [], []
        metadata = []
        for checkpoint in group:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            model = UtilityAbstentionGate(signal_dimension=signals.shape[1]).to(device).eval()
            model.load_state_dict(ckpt["model_state"])
            mean = np.asarray(ckpt["signal_mean"], dtype=np.float32)
            scale = np.asarray(ckpt["signal_scale"], dtype=np.float32)
            metadata.append(ckpt)
            ll, dd = [], []
            with torch.inference_mode():
                for start in range(0, len(signals), a.batch_size * 32):
                    sl = slice(start, start + a.batch_size * 32)
                    z = np.clip(((signals[sl] - mean) / scale).astype(np.float32), -5.0, 5.0)
                    logits, delta, _ = model(torch.from_numpy(z).to(device), torch.from_numpy(anchor[sl]).to(device))
                    ll.append(logits.cpu().numpy()); dd.append(delta.cpu().numpy())
            logits_all.append(np.concatenate(ll)); delta_all.append(np.concatenate(dd))
        use_retrieval = (np.mean(logits_all, axis=0) > 0.0) & (np.mean(delta_all, axis=0) > 0.0)
        errors = np.where(use_retrieval, r_errors, p_errors)
        rows.append({"checkpoints": [p.name for p in group], "outer_fold": int(metadata[0]["outer_fold"]), "retrieval_rate": float(use_retrieval.mean()), "metrics": summary(errors), "prototype": summary(p_errors), "retrieval": summary(r_errors)})
    payload = {"schema_version": 1, "official_test_evaluated": True, "method": "sv_ds_pa_mcsp_crossfitted_utility_official_ensemble" if a.ensemble else "sv_ds_pa_mcsp_crossfitted_utility_official", "device": str(device), "counts": {"train_gallery": int(len(train_idx)), "test_queries": int(len(test_idx)), "checkpoints": len(paths)}, "results": rows, "elapsed_seconds": time.perf_counter() - started}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
