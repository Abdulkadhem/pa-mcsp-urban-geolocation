#!/usr/bin/env python3
"""Validate a PA-MCSP scale-stage artifact set without opening official test."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidate-sweep", type=Path, required=True)
    parser.add_argument("--training-cache-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-allow-decision",
        default="ALLOW_25PCT",
        help="Scale-specific success label accepted by this validation run.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest)
    index = pd.read_parquet(args.feature_index)
    features = np.load(args.features, mmap_mode="r")
    feature_stats = json.loads(args.feature_stats.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    sweep = json.loads(args.candidate_sweep.read_text(encoding="utf-8"))
    nested = json.loads(args.training_cache_stats.read_text(encoding="utf-8"))
    checks = {
        "manifest_nonempty": len(manifest) > 0,
        "manifest_all_decoded": bool(manifest["decode_ok"].astype(bool).all()),
        "official_test_absent": not bool(
            manifest["official_split"].astype(str).eq("test").any()
        ),
        "feature_alignment": len(features) == len(index) == len(manifest),
        "feature_dimension": (
            features.ndim == 2
            and features.shape[1] == int(feature_stats["feature_dimension"])
        ),
        "feature_hash": (
            sha256_file(args.features) == feature_stats["features_sha256"]
        ),
        "index_hash": (
            sha256_file(args.feature_index) == feature_stats["index_sha256"]
        ),
        "metrics_test_closed": metrics["official_test_evaluated"] is False,
        "sweep_test_closed": sweep["official_test_evaluated"] is False,
        "nested_test_closed": nested["official_test_used"] is False,
        "three_seed_results": len(metrics["results"]) == 3,
        "decision_present": metrics["gate"]["decision"]
        in {args.expected_allow_decision, "REJECT_SCALE_UP"},
        "nested_block_overlap_zero": not any(
            int(value) for value in nested["overlap_audit"].values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"PA-MCSP validation failed: {failed}")
    payload = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "checks": checks,
        "counts": {
            "manifest": len(manifest),
            "train": int(manifest["split"].astype(str).eq("train").sum()),
            "development": int(
                manifest["split"].astype(str).eq("development").sum()
            ),
            "buffer_excluded": int(
                manifest["split"].astype(str).eq("buffer_excluded").sum()
            ),
        },
        "gate": metrics["gate"],
        "hashes": {
            "manifest_sha256": sha256_file(args.manifest),
            "features_sha256": sha256_file(args.features),
            "feature_index_sha256": sha256_file(args.feature_index),
            "metrics_sha256": sha256_file(args.metrics),
            "candidate_sweep_sha256": sha256_file(args.candidate_sweep),
            "training_cache_stats_sha256": sha256_file(
                args.training_cache_stats
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
