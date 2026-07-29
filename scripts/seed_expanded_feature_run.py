#!/usr/bin/env python3
"""Seed an expanded feature run from a completed subset without re-encoding it."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--old-stats", type=Path, required=True)
    parser.add_argument("--expanded-manifest", type=Path, required=True)
    parser.add_argument("--ordered-manifest-output", type=Path, required=True)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    old_index = pd.read_parquet(args.old_index)
    old_features = np.load(args.old_features, mmap_mode="r")
    old_stats = json.loads(args.old_stats.read_text(encoding="utf-8"))
    expanded = pd.read_parquet(args.expanded_manifest)
    expanded = expanded.loc[expanded["decode_ok"].astype(bool)].copy()

    if len(old_index) != len(old_features):
        raise RuntimeError("Old feature index and matrix have different lengths.")
    old_ids = old_index["id"].astype(str)
    expanded["id"] = expanded["id"].astype(str)
    if old_ids.duplicated().any() or expanded["id"].duplicated().any():
        raise RuntimeError("Feature seeding requires unique image IDs.")
    missing = set(old_ids) - set(expanded["id"])
    if missing:
        raise RuntimeError(
            f"Expanded manifest dropped {len(missing)} previously encoded IDs."
        )

    old_rank = pd.Series(
        np.arange(len(old_ids), dtype=np.int64),
        index=old_ids.to_numpy(),
    )
    expanded["_old_rank"] = expanded["id"].map(old_rank)
    reused = expanded.loc[expanded["_old_rank"].notna()].copy()
    reused["_old_rank"] = reused["_old_rank"].astype(np.int64)
    reused = reused.sort_values("_old_rank", kind="stable")
    if reused["id"].tolist() != old_ids.tolist():
        raise RuntimeError("Could not preserve the old feature row order.")
    new_rows = expanded.loc[expanded["_old_rank"].isna()].copy()
    sort_columns = [
        column
        for column in ("source_shard", "sequence", "id")
        if column in new_rows.columns
    ]
    new_rows = new_rows.sort_values(sort_columns, kind="stable")
    ordered = pd.concat((reused, new_rows), ignore_index=True)
    ordered = ordered.drop(columns=["_old_rank"])

    args.ordered_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.ordered_manifest_output.with_suffix(
        args.ordered_manifest_output.suffix + ".partial"
    )
    ordered.to_parquet(temporary_manifest, index=False)
    temporary_manifest.replace(args.ordered_manifest_output)
    manifest_hash = sha256_file(args.ordered_manifest_output)

    if args.state_output.is_file() and args.features_output.is_file():
        state = json.loads(args.state_output.read_text(encoding="utf-8"))
        if (
            state.get("manifest_sha256") == manifest_hash
            and int(state.get("total_rows", -1)) == len(ordered)
            and int(state.get("processed_rows", -1)) >= len(old_features)
        ):
            print(json.dumps({"status": "already_seeded", **state}, indent=2))
            return
        raise RuntimeError("An incompatible expanded feature state already exists.")

    dimension = int(old_features.shape[1])
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    seeded = np.lib.format.open_memmap(
        args.features_output,
        mode="w+",
        dtype=np.float16,
        shape=(len(ordered), dimension),
    )
    seeded[: len(old_features)] = old_features
    seeded.flush()
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "model_id": str(old_stats["model_id"]),
        "model_revision": str(old_stats["model_revision"]),
        "total_rows": len(ordered),
        "processed_rows": len(old_features),
        "reused_rows": len(old_features),
        "new_rows": len(ordered) - len(old_features),
        "feature_dimension": dimension,
        "feature_dtype": "float16",
        "batch_size": int(old_stats["batch_size"]),
        "num_workers": int(old_stats["num_workers"]),
        "active_seconds": float(old_stats["active_seconds"]),
        "features_output": str(args.features_output),
        "old_features_sha256": sha256_file(args.old_features),
    }
    atomic_json(args.state_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
