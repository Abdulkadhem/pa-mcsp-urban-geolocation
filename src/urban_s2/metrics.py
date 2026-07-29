"""Point-prediction metrics with a fixed denominator and explicit failures."""

from __future__ import annotations

import numpy as np

from .geodesy import EARTH_MEAN_RADIUS_KM, haversine_km

DEFAULT_THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0, 2500.0)


def evaluate_point_predictions(
    ground_truth_latlon: np.ndarray,
    predicted_latlon: np.ndarray,
    *,
    failure_mask: np.ndarray | None = None,
    thresholds_km: tuple[float, ...] = DEFAULT_THRESHOLDS_KM,
    failure_distance_km: float = np.pi * EARTH_MEAN_RADIUS_KM,
) -> dict[str, object]:
    """Evaluate all queries, counting invalid or declared failures as incorrect."""
    ground_truth = np.asarray(ground_truth_latlon, dtype=np.float64)
    predictions = np.asarray(predicted_latlon, dtype=np.float64)
    if ground_truth.ndim != 2 or ground_truth.shape[1] != 2:
        raise ValueError("ground_truth_latlon must have shape [n, 2]")
    if predictions.shape != ground_truth.shape:
        raise ValueError("predicted_latlon must match ground_truth_latlon")
    if any(threshold <= 0 for threshold in thresholds_km):
        raise ValueError("thresholds must be positive")

    valid_ground_truth = (
        np.isfinite(ground_truth).all(axis=1)
        & (np.abs(ground_truth[:, 0]) <= 90.0)
        & (np.abs(ground_truth[:, 1]) <= 180.0)
    )
    if not valid_ground_truth.all():
        raise ValueError("ground truth contains invalid coordinates")

    invalid_prediction = (
        ~np.isfinite(predictions).all(axis=1)
        | (np.abs(predictions[:, 0]) > 90.0)
        | (np.abs(predictions[:, 1]) > 180.0)
    )
    if failure_mask is None:
        failures = invalid_prediction
    else:
        failures = np.asarray(failure_mask, dtype=bool)
        if failures.shape != (ground_truth.shape[0],):
            raise ValueError("failure_mask must have shape [n]")
        failures = failures | invalid_prediction

    errors = np.full(ground_truth.shape[0], float(failure_distance_km), dtype=np.float64)
    successful = ~failures
    if successful.any():
        errors[successful] = haversine_km(
            ground_truth[successful, 0],
            ground_truth[successful, 1],
            predictions[successful, 0],
            predictions[successful, 1],
        )

    result: dict[str, object] = {
        "n_queries": int(ground_truth.shape[0]),
        "n_failures": int(failures.sum()),
        "failure_rate": float(failures.mean()) if failures.size else 0.0,
        "mean_error_km": float(np.mean(errors)),
        "median_error_km": float(np.median(errors)),
        "p75_error_km": float(np.percentile(errors, 75)),
        "p90_error_km": float(np.percentile(errors, 90)),
        "errors_km": errors,
    }
    result["accuracy"] = {
        str(int(threshold) if float(threshold).is_integer() else threshold): float(
            np.mean(errors <= threshold)
        )
        for threshold in thresholds_km
    }
    return result

