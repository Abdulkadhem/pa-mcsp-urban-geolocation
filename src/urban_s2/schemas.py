"""Immutable row contracts for data provenance and per-query predictions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

MANIFEST_REQUIRED_FIELDS = (
    "source_dataset",
    "source_revision",
    "image_id",
    "relative_path",
    "official_split",
    "development_split",
    "latitude",
    "longitude",
    "sequence_id",
    "ghsl_smod_class",
    "urban_flag",
    "decode_ok",
    "sha256",
    "license_spdx",
)

PREDICTION_REQUIRED_FIELDS = (
    "run_id",
    "query_id",
    "method",
    "seed",
    "official_split",
    "urban_flag",
    "ground_truth_latitude",
    "ground_truth_longitude",
    "predicted_latitude",
    "predicted_longitude",
    "error_km",
    "confidence",
    "latency_ms",
    "failure_code",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_fields(row: Mapping[str, object], required: tuple[str, ...]) -> list[str]:
    return [field for field in required if field not in row]


def _valid_coordinate(latitude: object, longitude: object) -> bool:
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude_value)
        and math.isfinite(longitude_value)
        and -90.0 <= latitude_value <= 90.0
        and -180.0 <= longitude_value <= 180.0
    )


def validate_manifest_row(row: Mapping[str, object], require_verified_hash: bool = True) -> list[str]:
    """Return validation errors for one manifest record."""
    errors: list[str] = []
    missing = _require_fields(row, MANIFEST_REQUIRED_FIELDS)
    if missing:
        return [f"missing field: {field}" for field in missing]

    if row["official_split"] not in {"train", "test"}:
        errors.append("official_split must be train or test")
    if row["development_split"] not in {"train", "validation", "calibration", "test"}:
        errors.append("development_split is invalid")
    if not _valid_coordinate(row["latitude"], row["longitude"]):
        errors.append("latitude/longitude are invalid")
    if not isinstance(row["urban_flag"], (bool, int)):
        errors.append("urban_flag must be boolean")
    if not isinstance(row["decode_ok"], (bool, int)):
        errors.append("decode_ok must be boolean")
    if not str(row["image_id"]).strip():
        errors.append("image_id must be non-empty")
    if not str(row["sequence_id"]).strip():
        errors.append("sequence_id must be non-empty")
    digest = str(row["sha256"]).lower()
    if require_verified_hash and not _SHA256.fullmatch(digest):
        errors.append("sha256 must be 64 lowercase hexadecimal characters")
    return errors


def validate_prediction_row(row: Mapping[str, object]) -> list[str]:
    """Return validation errors for one per-query prediction record."""
    errors: list[str] = []
    missing = _require_fields(row, PREDICTION_REQUIRED_FIELDS)
    if missing:
        return [f"missing field: {field}" for field in missing]

    if not _valid_coordinate(row["ground_truth_latitude"], row["ground_truth_longitude"]):
        errors.append("ground-truth coordinate is invalid")

    failure = str(row["failure_code"]).strip()
    prediction_valid = _valid_coordinate(row["predicted_latitude"], row["predicted_longitude"])
    if not failure and not prediction_valid:
        errors.append("successful predictions require a valid coordinate")

    for field in ("error_km", "latency_ms"):
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric")
            continue
        if not math.isfinite(value) or value < 0:
            errors.append(f"{field} must be finite and non-negative")

    try:
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            errors.append("confidence must lie in [0, 1]")
    except (TypeError, ValueError):
        errors.append("confidence must be numeric")
    return errors

