"""UrbanS²-Posterior research infrastructure."""

from .geodesy import (
    EARTH_MEAN_RADIUS_KM,
    angular_distance_rad,
    exp_map_s2,
    haversine_km,
    latlon_to_unit,
    spherical_mean,
    unit_to_latlon,
)
from .runtime import PilotMeasurement, RuntimeProjection, project_runtime
from .sampling import stable_group_fraction, stable_group_selected
from .schemas import (
    MANIFEST_REQUIRED_FIELDS,
    PREDICTION_REQUIRED_FIELDS,
    validate_manifest_row,
    validate_prediction_row,
)
from .vmf import log_vmf_normalizer_s2, vmf_log_prob, vmf_mixture_log_prob
from .metrics import evaluate_point_predictions

__all__ = [
    "EARTH_MEAN_RADIUS_KM",
    "MANIFEST_REQUIRED_FIELDS",
    "PREDICTION_REQUIRED_FIELDS",
    "PilotMeasurement",
    "RuntimeProjection",
    "angular_distance_rad",
    "evaluate_point_predictions",
    "exp_map_s2",
    "haversine_km",
    "latlon_to_unit",
    "log_vmf_normalizer_s2",
    "project_runtime",
    "spherical_mean",
    "stable_group_fraction",
    "stable_group_selected",
    "unit_to_latlon",
    "validate_manifest_row",
    "validate_prediction_row",
    "vmf_log_prob",
    "vmf_mixture_log_prob",
]
