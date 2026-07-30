"""Validate the small, public metadata for the locked PA-MCSP paper run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "run_id",
    "method",
    "backbone",
    "backbone_frozen",
    "trainable_checkpoint",
    "candidate_sources",
    "k_per_source",
    "patch_pooling",
    "score_normalization",
    "alpha",
    "official_queries",
    "official_metrics",
    "status",
}


def validate_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED.difference(payload)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if payload["trainable_checkpoint"] is not None:
        raise ValueError("the primary paper run must be learned-free")
    if payload["backbone_frozen"] is not True:
        raise ValueError("the primary backbone must be frozen")
    if payload["k_per_source"] != 16:
        raise ValueError("the locked paper run uses k=16 per source")
    if payload["alpha"] != 0.75:
        raise ValueError("the locked paper run uses alpha=0.75")
    metrics = payload["official_metrics"]
    for key in ("accuracy_25km", "accuracy_200km", "median_error_km", "mean_error_km", "geoscore"):
        if key not in metrics:
            raise ValueError(f"missing official metric: {key}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "pa_mcsp_paper_locked.json",
    )
    args = parser.parse_args()
    payload = validate_config(args.config)
    print(f"validated {payload['run_id']}: {payload['method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
