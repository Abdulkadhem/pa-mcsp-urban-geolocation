"""Prototype-anchored, candidate-aware spherical posterior head.

This module is intentionally independent from the original pilot head.  It
starts exactly at the frozen StreetCLIP prototype classifier and learns only a
residual scorer plus an optional, confidence-gated tangent refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .geodesy import EARTH_MEAN_RADIUS_KM


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors, dim=-1)


def spherical_exp_map(
    base: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    """Map tangent vectors to unit directions on S2."""
    base = normalize(base)
    tangent = tangent - (tangent * base).sum(dim=-1, keepdim=True) * base
    norm = tangent.norm(dim=-1, keepdim=True)
    # torch.sinc(x / pi) is sin(x) / x and has the correct derivative at 0.
    mapped = (
        torch.cos(norm) * base
        + torch.sinc(norm / math.pi) * tangent
    )
    return normalize(mapped)


def bounded_tangent(
    raw: torch.Tensor,
    base: torch.Tensor,
    *,
    maximum_radians: float,
) -> torch.Tensor:
    tangent = raw - (raw * base).sum(dim=-1, keepdim=True) * base
    norm = tangent.norm(dim=-1, keepdim=True)
    bounded_norm = maximum_radians * torch.tanh(norm)
    scale = torch.where(
        norm > 1e-8,
        bounded_norm / norm.clamp_min(1e-8),
        torch.full_like(norm, maximum_radians),
    )
    return tangent * scale


def angular_distance(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    return torch.acos(
        torch.clamp((normalize(first) * normalize(second)).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6)
    )


def log_vmf_normalizer(kappa: torch.Tensor) -> torch.Tensor:
    concentration = kappa.clamp_min(1e-4)
    log_sinh = (
        concentration
        - math.log(2.0)
        + torch.log1p(-torch.exp(-2.0 * concentration))
    )
    return (
        torch.log(concentration)
        - math.log(4.0 * math.pi)
        - log_sinh
    )


def force_gold_candidates(
    logits: torch.Tensor,
    gold_labels: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k labels/scores while guaranteeing inclusion of gold."""
    candidate_count = min(int(top_k), logits.shape[1])
    scores, labels = torch.topk(logits, k=candidate_count, dim=1)
    present = (labels == gold_labels[:, None]).any(dim=1)
    if not bool(present.all()):
        labels = labels.clone()
        scores = scores.clone()
        missing = ~present
        labels[missing, -1] = gold_labels[missing]
        scores[missing, -1] = logits[missing, gold_labels[missing]]
        order = torch.argsort(scores, dim=1, descending=True)
        labels = labels.gather(1, order)
        scores = scores.gather(1, order)
    return labels, scores


def mixture_vmf_nll(
    target_unit: torch.Tensor,
    means: torch.Tensor,
    kappas: torch.Tensor,
    log_weights: torch.Tensor,
) -> torch.Tensor:
    dots = torch.clamp(
        (target_unit[:, None, :] * means).sum(dim=-1),
        -1.0 + 1e-6,
        1.0 - 1e-6,
    )
    component_log_prob = (
        log_weights + log_vmf_normalizer(kappas) + kappas * dots
    )
    return -torch.logsumexp(component_log_prob, dim=1).mean()


@dataclass(frozen=True)
class HeadLoss:
    total: torch.Tensor
    listwise: torch.Tensor
    mixture_nll: torch.Tensor
    offset: torch.Tensor
    gate: torch.Tensor
    anchor: torch.Tensor
    natural_topk_recall: torch.Tensor
    forced_inclusion_rate: torch.Tensor


class PrototypeResidualHeadV2(nn.Module):
    """Residual scorer with an exact prototype-anchor initialization."""

    def __init__(
        self,
        *,
        feature_dimension: int,
        prototypes: torch.Tensor,
        centers_unit: torch.Tensor,
        leaf_counts: torch.Tensor,
        hidden_dimension: int = 96,
        temperature: float = 0.07,
        density_weight: float = 0.0,
        maximum_offset_radians: float = 0.12,
        enable_offsets: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("prototypes", normalize(prototypes.float()))
        self.register_buffer("centers_unit", normalize(centers_unit.float()))
        self.register_buffer(
            "log_leaf_counts",
            torch.log(leaf_counts.float().clamp_min(1)),
        )
        self.temperature = float(temperature)
        self.density_weight = float(density_weight)
        self.maximum_offset_radians = float(maximum_offset_radians)
        self.enable_offsets = bool(enable_offsets)

        self.residual_scorer = nn.Sequential(
            nn.Linear(feature_dimension + 1, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.residual_scorer[-1].weight)
        nn.init.zeros_(self.residual_scorer[-1].bias)

        candidate_dimension = feature_dimension * 2
        self.offset_head = nn.Sequential(
            nn.Linear(candidate_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 3),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        self.offset_gate_head = nn.Sequential(
            nn.Linear(candidate_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.offset_gate_head[-1].weight)
        nn.init.constant_(self.offset_gate_head[-1].bias, -2.0)
        self.kappa_head = nn.Sequential(
            nn.Linear(candidate_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.kappa_head[-1].weight)
        nn.init.constant_(self.kappa_head[-1].bias, 50.0)

    def score_components(
        self,
        features: torch.Tensor,
        anchor_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_features = normalize(features)
        if anchor_logits is None:
            anchor_logits = (
                base_features @ self.prototypes.T / self.temperature
            )
        density_logits = -self.density_weight * self.log_leaf_counts
        total_logits = anchor_logits + density_logits
        return {
            "anchor_logits": anchor_logits,
            "density_logits": density_logits,
            "total_logits": total_logits,
            "adapted_features": base_features,
        }

    def candidate_residual_scores(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = self.prototypes[labels]
        expanded_features = normalize(features)[:, None, :].expand_as(
            prototypes
        )
        cosine = (expanded_features * prototypes).sum(
            dim=-1,
            keepdim=True,
        )
        interactions = torch.cat(
            (expanded_features * prototypes, cosine),
            dim=-1,
        )
        return self.residual_scorer(interactions).squeeze(-1)

    def candidate_parameters(
        self,
        adapted_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidate_count = labels.shape[1]
        expanded_features = adapted_features[:, None, :].expand(
            -1, candidate_count, -1
        )
        prototypes = self.prototypes[labels]
        centers = self.centers_unit[labels]
        inputs = torch.cat((expanded_features, prototypes), dim=-1)
        gates = torch.sigmoid(self.offset_gate_head(inputs).squeeze(-1))
        if self.enable_offsets:
            tangent = bounded_tangent(
                self.offset_head(inputs),
                centers,
                maximum_radians=self.maximum_offset_radians,
            )
            means = spherical_exp_map(centers, gates[..., None] * tangent)
        else:
            means = centers
            gates = torch.zeros_like(gates)
        kappas = (
            F.softplus(self.kappa_head(inputs).squeeze(-1)) + 1e-3
        ).clamp_max(500.0)
        return {
            "means": means,
            "kappas": kappas,
            "gates": gates,
            "centers": centers,
        }

    def training_loss(
        self,
        features: torch.Tensor,
        gold_labels: torch.Tensor,
        target_unit: torch.Tensor,
        *,
        top_k: int,
        geo_scale_km: float,
        near_radius_km: float,
        mixture_weight: float,
        offset_weight: float,
        gate_weight: float,
        anchor_weight: float,
        anchor_logits: torch.Tensor | None = None,
    ) -> HeadLoss:
        components = self.score_components(features, anchor_logits)
        natural_count = min(int(top_k), components["total_logits"].shape[1])
        natural_labels = torch.topk(
            components["total_logits"],
            k=natural_count,
            dim=1,
        ).indices
        natural_recall = (
            natural_labels == gold_labels[:, None]
        ).any(dim=1).float().mean()
        labels, candidate_anchor_logits = force_gold_candidates(
            components["total_logits"],
            gold_labels,
            top_k,
        )
        candidate_residuals = self.candidate_residual_scores(
            components["adapted_features"],
            labels,
        )
        candidate_logits = candidate_anchor_logits + candidate_residuals
        parameters = self.candidate_parameters(
            components["adapted_features"],
            labels,
        )
        center_distances_km = (
            angular_distance(
                parameters["centers"],
                target_unit[:, None, :],
            )
            * EARTH_MEAN_RADIUS_KM
        )
        soft_targets = torch.softmax(
            -center_distances_km / float(geo_scale_km),
            dim=1,
        )
        log_weights = torch.log_softmax(candidate_logits, dim=1)
        listwise = -(soft_targets * log_weights).sum(dim=1).mean()
        mixture_nll = mixture_vmf_nll(
            target_unit,
            parameters["means"],
            parameters["kappas"],
            log_weights,
        )

        gold_candidates = labels == gold_labels[:, None]
        near = (center_distances_km <= float(near_radius_km)) | gold_candidates
        mean_angles = angular_distance(
            parameters["means"],
            target_unit[:, None, :],
        )
        near_weights = soft_targets * near.float()
        offset = (
            (mean_angles * near_weights).sum(dim=1)
            / near_weights.sum(dim=1).clamp_min(1e-8)
        ).mean()
        gate = F.binary_cross_entropy(
            parameters["gates"],
            near.float(),
        )
        anchor = candidate_residuals.square().mean()
        forced_inclusion_rate = 1.0 - natural_recall
        total = (
            listwise
            + float(mixture_weight) * mixture_nll
            + float(offset_weight) * offset
            + float(gate_weight) * gate
            + float(anchor_weight) * anchor
        )
        return HeadLoss(
            total=total,
            listwise=listwise,
            mixture_nll=mixture_nll,
            offset=offset,
            gate=gate,
            anchor=anchor,
            natural_topk_recall=natural_recall,
            forced_inclusion_rate=forced_inclusion_rate,
        )

    @torch.inference_mode()
    def posterior(
        self,
        features: torch.Tensor,
        *,
        top_k: int,
        anchor_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        components = self.score_components(features, anchor_logits)
        candidate_count = min(int(top_k), self.prototypes.shape[0])
        anchor_logits, labels = torch.topk(
            components["total_logits"],
            k=candidate_count,
            dim=1,
        )
        residual_logits = self.candidate_residual_scores(
            components["adapted_features"],
            labels,
        )
        logits = anchor_logits + residual_logits
        parameters = self.candidate_parameters(
            components["adapted_features"],
            labels,
        )
        weights = torch.softmax(logits, dim=1)
        map_indices = weights.argmax(dim=1)
        row_indices = torch.arange(len(features), device=features.device)
        map_point = parameters["means"][row_indices, map_indices]
        global_mean = normalize(
            (weights[..., None] * parameters["means"]).sum(dim=1)
        )
        anchor_labels = components["anchor_logits"].argmax(dim=1)
        return {
            "map_point": map_point,
            "global_mean": global_mean,
            "labels": labels,
            "weights": weights,
            "means": parameters["means"],
            "kappas": parameters["kappas"],
            "gates": parameters["gates"],
            "anchor_labels": anchor_labels,
            "anchor_logits": components["anchor_logits"],
            "residual_logits": residual_logits,
        }
