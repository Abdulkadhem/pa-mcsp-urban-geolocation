import numpy as np
import pytest

from urban_s2.vmf import log_vmf_normalizer_s2, vmf_log_prob, vmf_mixture_log_prob


def test_zero_concentration_is_uniform_on_sphere():
    expected = -np.log(4.0 * np.pi)
    assert log_vmf_normalizer_s2(0.0) == pytest.approx(expected)
    samples = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    mean = np.array([1.0, 0.0, 0.0])
    assert np.allclose(vmf_log_prob(samples, mean, 0.0), expected)


def test_mean_is_more_likely_than_antipode():
    mean = np.array([1.0, 0.0, 0.0])
    samples = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    log_prob = vmf_log_prob(samples, mean, 7.5)
    assert log_prob[0] - log_prob[1] == pytest.approx(15.0)


def test_large_concentration_is_finite():
    assert np.isfinite(log_vmf_normalizer_s2(np.array([50.0, 500.0, 5000.0]))).all()


def test_one_component_mixture_matches_component_density():
    samples = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    means = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    kappas = np.array([[2.0], [4.0]])
    weights = np.ones((2, 1))
    expected = vmf_log_prob(samples, means[:, 0, :], kappas[:, 0])
    actual = vmf_mixture_log_prob(samples, means, kappas, weights)
    assert np.allclose(actual, expected)


def test_negative_concentration_is_rejected():
    with pytest.raises(ValueError):
        log_vmf_normalizer_s2(-1.0)

