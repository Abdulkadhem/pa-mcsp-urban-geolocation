"""von Mises–Fisher probability utilities for distributions on S²."""

from __future__ import annotations

import numpy as np

from .geodesy import _normalize

_LOG_4PI = float(np.log(4.0 * np.pi))
_LOG_2 = float(np.log(2.0))


def log_vmf_normalizer_s2(kappa: np.ndarray | float) -> np.ndarray:
    """Return log C_3(kappa) for the vMF density on the two-sphere."""
    concentration = np.asarray(kappa, dtype=np.float64)
    if np.any(concentration < 0):
        raise ValueError("kappa must be non-negative")

    result = np.empty_like(concentration)
    small = concentration < 1e-4
    large = concentration > 20.0
    middle = ~(small | large)

    # log(k / sinh(k)) = -k²/6 + 7k⁴/360 + O(k⁶)
    k_small = concentration[small]
    result[small] = -_LOG_4PI - (k_small**2) / 6.0 + 7.0 * (k_small**4) / 360.0

    k_middle = concentration[middle]
    result[middle] = np.log(k_middle) - _LOG_4PI - np.log(np.sinh(k_middle))

    k_large = concentration[large]
    log_sinh = k_large - _LOG_2 + np.log1p(-np.exp(-2.0 * k_large))
    result[large] = np.log(k_large) - _LOG_4PI - log_sinh
    return result


def vmf_log_prob(samples: np.ndarray, mean_direction: np.ndarray, kappa: np.ndarray | float) -> np.ndarray:
    """Evaluate a vMF log density on S²."""
    sample_unit = _normalize(samples)
    mean_unit = _normalize(mean_direction)
    dot = np.sum(sample_unit * mean_unit, axis=-1)
    concentration = np.asarray(kappa, dtype=np.float64)
    return log_vmf_normalizer_s2(concentration) + concentration * dot


def vmf_mixture_log_prob(
    samples: np.ndarray,
    means: np.ndarray,
    kappas: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate one vMF mixture for each sample using stable log-sum-exp.

    Expected shapes are ``samples=[n, 3]``, ``means=[n, k, 3]``,
    ``kappas=[n, k]``, and ``weights=[n, k]``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    kappas = np.asarray(kappas, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("samples must have shape [n, 3]")
    if means.ndim != 3 or means.shape[:2] != kappas.shape or means.shape[2] != 3:
        raise ValueError("means must be [n, k, 3] and kappas must be [n, k]")
    if weights.shape != kappas.shape:
        raise ValueError("weights must match kappas")
    if samples.shape[0] != means.shape[0]:
        raise ValueError("sample and mixture batch dimensions must match")
    if np.any(weights < 0) or np.any(kappas < 0):
        raise ValueError("weights and kappas must be non-negative")

    weight_sums = np.sum(weights, axis=1, keepdims=True)
    if np.any(weight_sums <= 0):
        raise ValueError("each mixture must have positive total weight")
    normalized_weights = weights / weight_sums

    expanded_samples = samples[:, None, :]
    component_log_prob = vmf_log_prob(expanded_samples, means, kappas)
    component_log_prob += np.log(np.clip(normalized_weights, np.finfo(np.float64).tiny, None))
    maximum = np.max(component_log_prob, axis=1, keepdims=True)
    return np.squeeze(maximum, axis=1) + np.log(
        np.sum(np.exp(component_log_prob - maximum), axis=1)
    )

