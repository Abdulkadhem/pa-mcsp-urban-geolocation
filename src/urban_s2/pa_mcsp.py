"""Candidate generation and diagnostics for PA-MCSP.

The utilities in this module operate only on frozen image embeddings and
train-only geographic hypotheses.  They never inject the ground-truth
candidate, which keeps candidate coverage identical at training and inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .geodesy import EARTH_MEAN_RADIUS_KM, latlon_to_unit


@dataclass(frozen=True)
class CandidatePool:
    """Top-k candidate indices and cosine scores for each query."""

    indices: np.ndarray
    scores: np.ndarray
    seconds: float


class SafeResidualReranker(torch.nn.Module):
    """Small multi-cue residual reranker initialized at the anchor ranking."""

    def __init__(self, signal_dimension: int, hidden_dimension: int = 32) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(signal_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, 1),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        standardized_signals: torch.Tensor,
        anchor_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.network(standardized_signals).squeeze(-1)
        return anchor_logits + residual, residual


class SafeBinaryGate(torch.nn.Module):
    """Residual gate between the top prototype and retrieval hypotheses."""

    def __init__(self, signal_dimension: int, hidden_dimension: int = 24) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(signal_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, 1),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        standardized_signals: torch.Tensor,
        anchor_log_odds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.network(standardized_signals).squeeze(-1)
        return anchor_log_odds + residual, residual


class UtilityAbstentionGate(torch.nn.Module):
    """Two-head source utility model with a conservative residual prior.

    ``logit25`` is a residual binary utility logit around the raw source
    anchor. ``delta_log_error`` predicts the signed improvement in log1p
    geodesic error when selecting retrieval instead of the prototype.
    """

    def __init__(self, signal_dimension: int, hidden_dimension: int = 48) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(signal_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.GELU(),
        )
        self.logit25_head = torch.nn.Linear(hidden_dimension, 1)
        self.delta_head = torch.nn.Linear(hidden_dimension, 1)
        torch.nn.init.zeros_(self.logit25_head.weight)
        torch.nn.init.zeros_(self.logit25_head.bias)
        torch.nn.init.zeros_(self.delta_head.weight)
        torch.nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        standardized_signals: torch.Tensor,
        anchor_log_odds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(standardized_signals)
        residual = self.logit25_head(hidden).squeeze(-1)
        delta = self.delta_head(hidden).squeeze(-1)
        return anchor_log_odds + residual, delta, residual


def topk_cosine(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    top_k: int,
    batch_size: int = 128,
) -> CandidatePool:
    """Return exact top-k cosine candidates with bounded query-batch memory."""
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("query and gallery features must be matrices")
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("query and gallery feature dimensions differ")
    if len(query_features) == 0 or len(gallery_features) == 0:
        raise ValueError("query and gallery features must be non-empty")
    candidate_count = min(int(top_k), len(gallery_features))
    if candidate_count <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")

    gallery = F.normalize(gallery_features.float(), dim=1)
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(query_features), int(batch_size)):
            query = F.normalize(
                query_features[start : start + int(batch_size)].float(),
                dim=1,
            )
            scores, indices = torch.topk(
                query @ gallery.T,
                k=candidate_count,
                dim=1,
                sorted=True,
            )
            all_indices.append(indices.cpu().numpy().astype(np.int32))
            all_scores.append(scores.cpu().numpy().astype(np.float32))
    return CandidatePool(
        indices=np.concatenate(all_indices, axis=0),
        scores=np.concatenate(all_scores, axis=0),
        seconds=time.perf_counter() - started,
    )


def candidate_errors_km(
    truth_latlon: np.ndarray,
    candidate_latlon: np.ndarray,
) -> np.ndarray:
    """Return great-circle errors for an ``N × K`` candidate array."""
    truth = np.asarray(truth_latlon, dtype=np.float64)
    candidates = np.asarray(candidate_latlon, dtype=np.float64)
    if truth.shape != (len(truth), 2):
        raise ValueError("truth_latlon must have shape (N, 2)")
    if candidates.ndim != 3 or candidates.shape[0] != len(truth):
        raise ValueError("candidate_latlon must have shape (N, K, 2)")
    truth_unit = latlon_to_unit(truth[:, 0], truth[:, 1])
    candidate_unit = latlon_to_unit(
        candidates[..., 0],
        candidates[..., 1],
    )
    dots = np.einsum(
        "nd,nkd->nk",
        truth_unit,
        candidate_unit,
        optimize=True,
    )
    return (
        np.arccos(np.clip(dots, -1.0, 1.0))
        * EARTH_MEAN_RADIUS_KM
    )


def oracle_summary(errors_km: np.ndarray) -> dict[str, float | int]:
    """Summarize the best attainable candidate error for each query."""
    errors = np.asarray(errors_km, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] == 0:
        raise ValueError("errors_km must be a non-empty matrix")
    best = errors.min(axis=1)
    return {
        "queries": int(len(best)),
        "candidates_per_query": int(errors.shape[1]),
        "accuracy_1km": float(np.mean(best <= 1.0)),
        "accuracy_25km": float(np.mean(best <= 25.0)),
        "accuracy_200km": float(np.mean(best <= 200.0)),
        "median_error_km": float(np.median(best)),
        "p75_error_km": float(np.quantile(best, 0.75)),
        "p90_error_km": float(np.quantile(best, 0.90)),
    }


def paired_union_oracle(
    prototype_errors: np.ndarray,
    retrieval_errors: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    """Summarize a natural union of the first K candidates from each source."""
    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be positive")
    if prototype_errors.shape[0] != retrieval_errors.shape[0]:
        raise ValueError("candidate pools have different query counts")
    combined = np.concatenate(
        (prototype_errors[:, :k], retrieval_errors[:, :k]),
        axis=1,
    )
    summary = oracle_summary(combined)
    summary["prototype_k"] = min(k, prototype_errors.shape[1])
    summary["retrieval_k"] = min(k, retrieval_errors.shape[1])
    return summary


def build_multicue_signals(
    prototype_scores: np.ndarray,
    retrieval_scores: np.ndarray,
    prototype_latlon: np.ndarray,
    retrieval_latlon: np.ndarray,
    prototype_density: np.ndarray,
    retrieval_density: np.ndarray,
    *,
    anchor_temperature: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Build PA-MCSP signals and source-comparable anchor logits.

    Candidate order is prototype candidates followed by retrieval candidates.
    All signals are available at inference time.
    """
    prototype_scores = np.asarray(prototype_scores, dtype=np.float32)
    retrieval_scores = np.asarray(retrieval_scores, dtype=np.float32)
    prototype_latlon = np.asarray(prototype_latlon, dtype=np.float64)
    retrieval_latlon = np.asarray(retrieval_latlon, dtype=np.float64)
    prototype_density = np.asarray(prototype_density, dtype=np.float32)
    retrieval_density = np.asarray(retrieval_density, dtype=np.float32)
    if prototype_scores.shape != retrieval_scores.shape:
        raise ValueError("prototype and retrieval score shapes must match")
    n_queries, top_k = prototype_scores.shape
    expected_coordinate_shape = (n_queries, top_k, 2)
    if (
        prototype_latlon.shape != expected_coordinate_shape
        or retrieval_latlon.shape != expected_coordinate_shape
    ):
        raise ValueError("candidate coordinates have an unexpected shape")
    if (
        prototype_density.shape != prototype_scores.shape
        or retrieval_density.shape != retrieval_scores.shape
    ):
        raise ValueError("candidate density shapes must match scores")
    if anchor_temperature <= 0.0:
        raise ValueError("anchor_temperature must be positive")

    scores = np.concatenate((prototype_scores, retrieval_scores), axis=1)
    own_top = np.concatenate(
        (
            np.repeat(prototype_scores[:, :1], top_k, axis=1),
            np.repeat(retrieval_scores[:, :1], top_k, axis=1),
        ),
        axis=1,
    )
    own_mean = np.concatenate(
        (
            np.repeat(prototype_scores.mean(1, keepdims=True), top_k, axis=1),
            np.repeat(retrieval_scores.mean(1, keepdims=True), top_k, axis=1),
        ),
        axis=1,
    )
    own_std = np.concatenate(
        (
            np.repeat(
                prototype_scores.std(1, keepdims=True) + 1e-6,
                top_k,
                axis=1,
            ),
            np.repeat(
                retrieval_scores.std(1, keepdims=True) + 1e-6,
                top_k,
                axis=1,
            ),
        ),
        axis=1,
    )
    ranks = np.tile(
        np.arange(1, top_k + 1, dtype=np.float32),
        (n_queries, 1),
    )
    reciprocal_rank = np.concatenate((1.0 / ranks, 1.0 / ranks), axis=1)
    source = np.concatenate(
        (
            np.zeros_like(prototype_scores),
            np.ones_like(retrieval_scores),
        ),
        axis=1,
    )
    density = np.concatenate(
        (prototype_density, retrieval_density),
        axis=1,
    )

    prototype_unit = latlon_to_unit(
        prototype_latlon[..., 0],
        prototype_latlon[..., 1],
    )
    retrieval_unit = latlon_to_unit(
        retrieval_latlon[..., 0],
        retrieval_latlon[..., 1],
    )
    cross_dots = np.einsum(
        "nkd,njd->nkj",
        prototype_unit,
        retrieval_unit,
        optimize=True,
    )
    cross_distance = (
        np.arccos(np.clip(cross_dots, -1.0, 1.0))
        * EARTH_MEAN_RADIUS_KM
    )
    cross_min = np.concatenate(
        (cross_distance.min(axis=2), cross_distance.min(axis=1)),
        axis=1,
    )
    support_25 = np.concatenate(
        (
            (cross_distance <= 25.0).mean(axis=2),
            (cross_distance <= 25.0).mean(axis=1),
        ),
        axis=1,
    )
    support_100 = np.concatenate(
        (
            (cross_distance <= 100.0).mean(axis=2),
            (cross_distance <= 100.0).mean(axis=1),
        ),
        axis=1,
    )
    support_200 = np.concatenate(
        (
            (cross_distance <= 200.0).mean(axis=2),
            (cross_distance <= 200.0).mean(axis=1),
        ),
        axis=1,
    )
    coordinates = np.concatenate((prototype_latlon, retrieval_latlon), axis=1)
    top_prototype = np.repeat(prototype_latlon[:, :1], 2 * top_k, axis=1)
    top_retrieval = np.repeat(retrieval_latlon[:, :1], 2 * top_k, axis=1)
    distance_to_top_prototype = candidate_errors_km(
        top_prototype[:, 0],
        coordinates,
    )
    distance_to_top_retrieval = candidate_errors_km(
        top_retrieval[:, 0],
        coordinates,
    )

    signals = np.stack(
        (
            scores,
            scores - own_top,
            (scores - own_mean) / own_std,
            reciprocal_rank,
            source,
            np.log1p(density),
            np.log1p(distance_to_top_prototype),
            np.log1p(distance_to_top_retrieval),
            np.log1p(cross_min),
            support_25,
            support_100,
            support_200,
        ),
        axis=2,
    ).astype(np.float32)
    anchor_logits = (
        (scores - scores.max(axis=1, keepdims=True))
        / float(anchor_temperature)
    ).astype(np.float32)
    return signals, anchor_logits


def reranking_loss(
    logits: torch.Tensor,
    residuals: torch.Tensor,
    candidate_errors: torch.Tensor,
    *,
    expected_error_weight: float = 0.2,
    anchor_weight: float = 0.05,
) -> torch.Tensor:
    """Natural-candidate listwise loss aligned to geodesic point selection."""
    best_candidate = candidate_errors.argmin(dim=1)
    classification = F.cross_entropy(logits, best_candidate)
    weights = torch.softmax(logits, dim=1)
    scaled_error = torch.log1p(candidate_errors)
    expected_error = (weights * scaled_error).sum(dim=1).mean()
    anchor = residuals.square().mean()
    return (
        classification
        + float(expected_error_weight) * expected_error
        + float(anchor_weight) * anchor
    )
