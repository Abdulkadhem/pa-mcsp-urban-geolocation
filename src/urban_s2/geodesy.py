"""Numerically stable spherical geometry used by training and evaluation."""

from __future__ import annotations

import numpy as np

EARTH_MEAN_RADIUS_KM = 6371.0088
_EPS = np.finfo(np.float64).eps


def latlon_to_unit(latitude_deg: np.ndarray | float, longitude_deg: np.ndarray | float) -> np.ndarray:
    """Convert WGS84-style latitude/longitude degrees to unit vectors on S²."""
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    latitude, longitude = np.broadcast_arrays(latitude, longitude)
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError("zero-length vectors cannot be normalized")
    return vectors / norms


def unit_to_latlon(unit_vectors: np.ndarray) -> np.ndarray:
    """Convert vectors on S² to ``[..., latitude, longitude]`` degrees."""
    unit = _normalize(unit_vectors)
    latitude = np.arctan2(unit[..., 2], np.hypot(unit[..., 0], unit[..., 1]))
    longitude = np.arctan2(unit[..., 1], unit[..., 0])
    return np.stack((np.rad2deg(latitude), np.rad2deg(longitude)), axis=-1)


def angular_distance_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Great-circle angle using atan2(||a×b||, a·b), stable near 0 and π."""
    first_unit = _normalize(first)
    second_unit = _normalize(second)
    cross_norm = np.linalg.norm(np.cross(first_unit, second_unit), axis=-1)
    dot = np.sum(first_unit * second_unit, axis=-1)
    return np.arctan2(cross_norm, np.clip(dot, -1.0, 1.0))


def haversine_km(
    first_latitude_deg: np.ndarray | float,
    first_longitude_deg: np.ndarray | float,
    second_latitude_deg: np.ndarray | float,
    second_longitude_deg: np.ndarray | float,
    radius_km: float = EARTH_MEAN_RADIUS_KM,
) -> np.ndarray:
    """Benchmark-compatible spherical distance in kilometres."""
    first = latlon_to_unit(first_latitude_deg, first_longitude_deg)
    second = latlon_to_unit(second_latitude_deg, second_longitude_deg)
    return radius_km * angular_distance_rad(first, second)


def exp_map_s2(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Map a tangent vector at ``base`` back to S² with the exponential map."""
    base_unit = _normalize(base)
    tangent = np.asarray(tangent, dtype=np.float64)
    if base_unit.shape != tangent.shape:
        base_unit, tangent = np.broadcast_arrays(base_unit, tangent)

    # Remove any numerical radial component so that tangent lies in T_base S².
    tangent = tangent - np.sum(tangent * base_unit, axis=-1, keepdims=True) * base_unit
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    direction = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-12)
    mapped = np.cos(norm) * base_unit + np.sin(norm) * direction
    mapped = np.where(norm > 1e-12, mapped, base_unit)
    return _normalize(mapped)


def spherical_mean(unit_vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted extrinsic mean on S², normalized back to the sphere."""
    unit = _normalize(unit_vectors)
    if unit.ndim != 2 or unit.shape[1] != 3:
        raise ValueError("unit_vectors must have shape [n, 3]")
    if weights is None:
        weights_array = np.ones(unit.shape[0], dtype=np.float64)
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (unit.shape[0],):
            raise ValueError("weights must have shape [n]")
        if np.any(weights_array < 0) or not np.any(weights_array > 0):
            raise ValueError("weights must be non-negative with at least one positive value")
    resultant = np.sum(unit * weights_array[:, None], axis=0)
    if np.linalg.norm(resultant) <= 1e-10:
        raise ValueError("spherical mean is undefined for a near-zero resultant")
    return _normalize(resultant)

