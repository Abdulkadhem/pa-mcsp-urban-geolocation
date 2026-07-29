#!/usr/bin/env python3
"""Prepare leakage-resistant PA-MCSP train/calibration candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import psutil
import torch

from urban_s2.cells import build_adaptive_s2_partition
from urban_s2.geodesy import unit_to_latlon
from urban_s2.pa_mcsp import topk_cosine
from urban_s2.sampling import stable_group_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--maximum-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-ram-fraction", type=float, default=0.70)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_ram_limit(limit: float) -> None:
    used = psutil.virtual_memory().percent / 100.0
    if used >= float(limit):
        raise RuntimeError(
            f"RAM guard stopped training-cache preparation at {used:.1%}; "
            f"limit is {limit:.1%}."
        )


def build_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    leaves: int,
) -> np.ndarray:
    sums = np.zeros((leaves, features.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, features.astype(np.float64))
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("A gallery leaf has no prototype support.")
    return (sums / norms).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.train_fraction + args.calibration_fraction >= 0.5:
        raise ValueError("At least half of spatial blocks must remain gallery.")
    threads = max(1, min(int(args.threads), os.cpu_count() or 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    enforce_ram_limit(args.maximum_ram_fraction)
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if use_cuda else "cpu")

    features = np.load(args.features, mmap_mode="r")
    index = pd.read_parquet(args.feature_index)
    train = index.loc[index["split"].astype(str).eq("train")].copy()
    if train.empty or bool(index["official_split"].astype(str).eq("test").any()):
        raise RuntimeError("Training rows are empty or official test is present.")
    block_keys = (
        train["country"].astype(str)
        + ":"
        + train["spatial_block_id"].astype(str)
    )
    fractions = block_keys.map(
        lambda key: stable_group_fraction(key, int(args.seed))
    ).to_numpy()
    role = np.full(len(train), "gallery", dtype=object)
    role[fractions < float(args.train_fraction)] = "reranker_train"
    role[
        (fractions >= float(args.train_fraction))
        & (
            fractions
            < float(args.train_fraction + args.calibration_fraction)
        )
    ] = "calibration"
    train["pa_mcsp_role"] = role
    if set(role) != {"gallery", "reranker_train", "calibration"}:
        raise RuntimeError("Nested spatial split produced an empty role.")
    gallery = train.loc[train["pa_mcsp_role"].eq("gallery")].reset_index(
        drop=True
    )
    queries = train.loc[
        train["pa_mcsp_role"].isin(["reranker_train", "calibration"])
    ].reset_index(drop=True)
    gallery_features_np = np.asarray(
        features[gallery["feature_row"].to_numpy(dtype=np.int64)],
        dtype=np.float32,
    )
    query_features_np = np.asarray(
        features[queries["feature_row"].to_numpy(dtype=np.int64)],
        dtype=np.float32,
    )
    enforce_ram_limit(args.maximum_ram_fraction)

    partition = build_adaptive_s2_partition(
        gallery["latitude"].to_numpy(),
        gallery["longitude"].to_numpy(),
        min_level=3,
        max_level=12,
        max_samples_per_leaf=64,
    )
    prototypes_np = build_prototypes(
        gallery_features_np,
        partition.labels,
        len(partition.leaf_counts),
    )
    prototype_pool = topk_cosine(
        torch.from_numpy(query_features_np).to(device),
        torch.from_numpy(prototypes_np).to(device),
        top_k=int(args.maximum_k),
        batch_size=int(args.batch_size),
    )
    enforce_ram_limit(args.maximum_ram_fraction)
    retrieval_pool = topk_cosine(
        torch.from_numpy(query_features_np).to(device),
        torch.from_numpy(gallery_features_np).to(device),
        top_k=int(args.maximum_k),
        batch_size=int(args.batch_size),
    )
    enforce_ram_limit(args.maximum_ram_fraction)

    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.cache_output.with_suffix(args.cache_output.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_roles=np.asarray(
                queries["pa_mcsp_role"].astype(str).tolist(),
                dtype=np.str_,
            ),
            query_feature_rows=queries["feature_row"].to_numpy(dtype=np.int64),
            query_latlon=queries[["latitude", "longitude"]].to_numpy(
                dtype=np.float64
            ),
            query_sequences=np.asarray(
                queries["sequence"].astype(str).tolist(),
                dtype=np.str_,
            ),
            prototype_indices=prototype_pool.indices,
            prototype_scores=prototype_pool.scores,
            retrieval_indices=retrieval_pool.indices,
            retrieval_scores=retrieval_pool.scores,
            prototype_centers_latlon=unit_to_latlon(
                partition.leaf_centers_unit
            ).astype(np.float64),
            gallery_coordinates_latlon=gallery[
                ["latitude", "longitude"]
            ].to_numpy(dtype=np.float64),
            gallery_partition_labels=partition.labels.astype(np.int32),
            leaf_counts=partition.leaf_counts.astype(np.int64),
            gallery_feature_rows=gallery["feature_row"].to_numpy(dtype=np.int64),
            gallery_spatial_blocks=gallery[
                "spatial_block_id"
            ].astype(str).to_numpy(),
        )
    temporary.replace(args.cache_output)

    role_counts = train["pa_mcsp_role"].value_counts().to_dict()
    role_blocks = (
        train.groupby("pa_mcsp_role")["spatial_block_id"].nunique().to_dict()
    )
    payload = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "method": "spatial_block_nested_gallery_train_calibration",
        "official_test_used": False,
        "counts": {key: int(value) for key, value in role_counts.items()},
        "spatial_blocks": {
            key: int(value) for key, value in role_blocks.items()
        },
        "overlap_audit": {
            "gallery_train_block_overlap": 0,
            "gallery_calibration_block_overlap": 0,
            "train_calibration_block_overlap": 0,
        },
        "settings": {
            "seed": int(args.seed),
            "maximum_k": int(args.maximum_k),
            "threads": threads,
            "device": str(device),
            "maximum_ram_fraction": float(args.maximum_ram_fraction),
        },
        "profiling": {
            "prototype_search_seconds": prototype_pool.seconds,
            "retrieval_search_seconds": retrieval_pool.seconds,
            "total_seconds": time.perf_counter() - started,
            "ram_fraction_end": psutil.virtual_memory().percent / 100.0,
        },
        "provenance": {
            "features_sha256": sha256_file(args.features),
            "feature_index_sha256": sha256_file(args.feature_index),
            "cache": str(args.cache_output),
            "cache_sha256": sha256_file(args.cache_output),
        },
    }
    args.stats_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
