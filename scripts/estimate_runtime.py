#!/usr/bin/env python3
"""CLI for projecting full-run duration and cost from pilot timing JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_s2.runtime import PilotMeasurement, project_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True, help="Pilot timing JSON.")
    parser.add_argument("--target-train-images", type=int, default=4_894_685)
    parser.add_argument("--target-test-images", type=int, default=210_122)
    parser.add_argument("--target-epochs", type=float, default=5.0)
    parser.add_argument("--target-seeds", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--scaling-efficiency", type=float, default=0.85)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--gpu-hourly-price", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.pilot.read_text(encoding="utf-8"))
    measurement = PilotMeasurement(**payload)
    projection = project_runtime(
        measurement,
        target_train_images=args.target_train_images,
        target_test_images=args.target_test_images,
        target_epochs=args.target_epochs,
        target_seeds=args.target_seeds,
        gpu_count=args.gpus,
        scaling_efficiency=args.scaling_efficiency,
        safety_factor=args.safety_factor,
        gpu_hourly_price_usd=args.gpu_hourly_price,
    )
    rendered = json.dumps(projection.to_dict(), indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

