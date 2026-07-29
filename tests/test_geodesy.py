import numpy as np
import pytest

from urban_s2.geodesy import (
    EARTH_MEAN_RADIUS_KM,
    angular_distance_rad,
    exp_map_s2,
    haversine_km,
    latlon_to_unit,
    spherical_mean,
    unit_to_latlon,
)


def test_latlon_round_trip_including_dateline_and_poles():
    coordinates = np.array(
        [[0.0, 0.0], [33.3152, 44.3661], [-45.0, 179.999], [90.0, 0.0]]
    )
    reconstructed = unit_to_latlon(latlon_to_unit(coordinates[:, 0], coordinates[:, 1]))
    assert np.allclose(reconstructed[:, 0], coordinates[:, 0], atol=1e-9)
    assert np.allclose(reconstructed[:-1, 1], coordinates[:-1, 1], atol=1e-9)


def test_antipodes_have_pi_angular_distance():
    first = latlon_to_unit(0.0, 0.0)
    second = latlon_to_unit(0.0, 180.0)
    assert angular_distance_rad(first, second) == pytest.approx(np.pi)
    assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(
        np.pi * EARTH_MEAN_RADIUS_KM
    )


def test_exp_map_moves_along_equator():
    base = np.array([1.0, 0.0, 0.0])
    tangent = np.array([0.0, np.pi / 2.0, 0.0])
    mapped = exp_map_s2(base, tangent)
    assert np.allclose(mapped, [0.0, 1.0, 0.0], atol=1e-10)


def test_exp_map_zero_is_identity():
    base = latlon_to_unit(33.3152, 44.3661)
    assert np.allclose(exp_map_s2(base, np.zeros(3)), base)


def test_weighted_spherical_mean_prefers_heavier_point():
    points = latlon_to_unit([0.0, 0.0], [0.0, 90.0])
    mean = spherical_mean(points, weights=np.array([3.0, 1.0]))
    latitude, longitude = unit_to_latlon(mean)
    assert latitude == pytest.approx(0.0)
    assert 0.0 < longitude < 45.0


def test_spherical_mean_rejects_antipodal_resultant():
    with pytest.raises(ValueError):
        spherical_mean(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

