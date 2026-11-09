#!/usr/bin/env python3
"""Nested, fold-isolated evaluation of StreetCLIP patch-token late fusion.

The fusion weight is selected only on non-outer folds.  The outer fold is
then evaluated once, avoiding the optimistic pilot protocol.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def geo_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lat1 = np.radians(a[..., 0]); lon1 = np.radians(a[..., 1])
    lat2 = np.radians(b[..., 0]); lon2 = np.radians(b[..., 1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def metrics(err: np.ndarray) -> dict:
    return {
        "n": int(len(err)),
        "acc25": float(np.mean(err <= 25.0)),
        "median_km": float(np.median(err)),
        "geoscore": float(np.mean(np.exp(-err / 1492.7))),
    }


def lookup_rows(desc_rows: np.ndarray, rows: np.ndarray, what: str) -> np.ndarray:
    """Exact positions of ``rows`` in the sorted ``desc_rows``.

    ``np.searchsorted`` alone returns an insertion point for absent rows, which
    would silently read the descriptor of a different image. O(n log m).
    """
    pos = np.searchsorted(desc_rows, rows)
    clipped = np.minimum(pos, len(desc_rows) - 1)
    present = desc_rows[clipped] == rows
    if not present.all():
        missing = np.unique(np.asarray(rows)[~present])
        raise KeyError(
            f"{missing.size} {what} feature rows are absent from the descriptor "
            f"cache (first: {missing[:5].tolist()}); encode them before scoring."
        )
    return clipped


def score_rows(rows, cache, folds, desc, desc_rows, k, batch, device):
    pidx = cache["prototype_indices"][:,:k].astype(np.int64)
    ridx = cache["retrieval_indices"][:,:k].astype(np.int64)
    ps = cache["prototype_scores"][:,:k].astype(np.float32)
    rs = cache["retrieval_scores"][:,:k].astype(np.float32)
    qrows = cache["query_feature_rows"].astype(np.int64)
    grows = cache["gallery_feature_rows"].astype(np.int64)
    glabel = cache["gallery_partition_labels"].astype(np.int32)
    qc = cache["query_latlon"].astype(np.float64)
    pc = cache["prototype_centers_latlon"].astype(np.float64)
    gc = cache["gallery_coordinates_latlon"].astype(np.float64)
    reps = np.full(len(cache["leaf_counts"]), -1, np.int64)
    seen = np.zeros(len(reps), bool)
    for i, g in enumerate(glabel):
        if not seen[g]: reps[g] = grows[i]; seen[g] = True
    rows = np.asarray(rows, dtype=np.int64)
    raw = np.empty((len(rows), 2*k), np.float32)
    patch = np.empty_like(raw)
    err = np.empty_like(raw)
    with torch.inference_mode():
      for start in range(0, len(rows), batch):
        rr = rows[start:start+batch]
        cand = np.concatenate((reps[pidx[rr]], grows[ridx[rr]]), axis=1)
        qpos = lookup_rows(desc_rows, qrows[rr], "query")
        cpos = lookup_rows(desc_rows, cand, "candidate")
        qv = torch.from_numpy(np.asarray(desc[qpos])).to(device=device, non_blocking=True)
        cv = torch.from_numpy(np.asarray(desc[cpos])).to(device=device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            patch[start:start+len(rr)] = (qv[:, None] * cv).sum(-1).mean(-1).float().cpu().numpy()
        raw[start:start+len(rr)] = np.concatenate((ps[rr], rs[rr]), axis=1)
        coords = np.concatenate((pc[pidx[rr]], gc[ridx[rr]]), axis=1)
        err[start:start+len(rr)] = geo_km(qc[rr, None, :], coords)
    return raw, patch, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--descriptor", type=Path, required=True)
    ap.add_argument("--descriptor-rows", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cache = np.load(args.cache, allow_pickle=True)
    folds = pd.read_parquet(args.folds).sort_values("query_position").reset_index(drop=True)
    desc_rows = np.load(args.descriptor_rows, mmap_mode="r").astype(np.int64, copy=False)
    desc = np.lib.format.open_memmap(args.descriptor, mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_ids = np.sort(folds.fold.unique().astype(int))
    all_idx = np.arange(len(folds), dtype=np.int64)
    results = []
    for fold in fold_ids:
        outer = all_idx[folds.fold.to_numpy(np.int64) == fold]
        calib = all_idx[folds.fold.to_numpy(np.int64) != fold]
        raw_c, patch_c, err_c = score_rows(calib, cache, folds, desc, desc_rows, args.k, args.batch, device)
        raw_o, patch_o, err_o = score_rows(outer, cache, folds, desc, desc_rows, args.k, args.batch, device)
        def z(x): return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-6)
        rz_c, pz_c = z(raw_c), z(patch_c)
        rz_o, pz_o = z(raw_o), z(patch_o)
        grid = [0.0, .05, .1, .15, .2, .3, .5, .75, 1.0]
        choices = []
        for alpha in grid:
            pred = np.argmax(rz_c + alpha*pz_c, 1)
            choices.append((metrics(err_c[np.arange(len(err_c)), pred]), alpha))
        choices.sort(key=lambda x: (-x[0]["acc25"], x[0]["median_km"], -x[0]["geoscore"]))
        alpha = choices[0][1]
        def eval_scores(rz, pz, er):
            out = {}
            for name, score in [("prototype", rz[:, :args.k]), ("retrieval", rz[:, args.k:]),
                                ("union_anchor", rz), ("patch_fusion", rz + alpha*pz)]:
                pred = np.argmax(score, 1)
                out[name] = metrics(er[np.arange(len(er)), pred])
            return out
        results.append({"fold": int(fold), "outer_n": int(len(outer)), "calibration_n": int(len(calib)),
                        "selected_alpha": float(alpha), "calibration_selection": choices[0][0],
                        "outer": eval_scores(rz_o, pz_o, err_o)})
        print(json.dumps(results[-1]), flush=True)
    def mean_metric(name, key): return float(np.mean([r["outer"][name][key] for r in results]))
    summary = {"method": "nested_patch_token_late_fusion", "folds": results,
               "mean_outer": {n: {m: mean_metric(n, m) for m in ("acc25", "median_km", "geoscore")}
                              for n in ("prototype", "retrieval", "union_anchor", "patch_fusion")},
               "elapsed_seconds": time.perf_counter()-t0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
