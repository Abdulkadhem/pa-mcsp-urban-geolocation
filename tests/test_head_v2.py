from __future__ import annotations

import numpy as np
import torch

from urban_s2.head_v2 import (
    PrototypeResidualHeadV2,
    force_gold_candidates,
)


def make_model(*, enable_offsets: bool = True) -> PrototypeResidualHeadV2:
    generator = torch.Generator().manual_seed(11)
    prototypes = torch.nn.functional.normalize(
        torch.randn(5, 12, generator=generator),
        dim=1,
    )
    centers = torch.nn.functional.normalize(
        torch.randn(5, 3, generator=generator),
        dim=1,
    )
    return PrototypeResidualHeadV2(
        feature_dimension=12,
        prototypes=prototypes,
        centers_unit=centers,
        leaf_counts=torch.tensor([2, 3, 5, 7, 11]),
        hidden_dimension=8,
        density_weight=0.0,
        enable_offsets=enable_offsets,
    )


def test_v2_initialization_is_exactly_prototype_anchored() -> None:
    model = make_model()
    features = torch.nn.functional.normalize(torch.randn(4, 12), dim=1)
    scores = model.score_components(features)

    assert torch.equal(
        scores["total_logits"],
        scores["anchor_logits"],
    )
    assert torch.count_nonzero(scores["density_logits"]) == 0
    labels = scores["anchor_logits"].topk(k=3, dim=1).indices
    assert torch.count_nonzero(
        model.candidate_residual_scores(features, labels)
    ) == 0


def test_forced_candidate_set_always_contains_gold() -> None:
    logits = torch.tensor(
        [
            [9.0, 8.0, 7.0, 1.0],
            [1.0, 3.0, 2.0, 4.0],
        ]
    )
    gold = torch.tensor([3, 1])
    labels, scores = force_gold_candidates(logits, gold, top_k=2)

    assert labels.shape == scores.shape == (2, 2)
    assert torch.all((labels == gold[:, None]).any(dim=1))


def test_map_decoder_returns_a_component_not_the_global_mean() -> None:
    model = make_model(enable_offsets=False)
    features = torch.nn.functional.normalize(torch.randn(3, 12), dim=1)
    posterior = model.posterior(features, top_k=3)
    selected = posterior["means"][:, 0]

    assert torch.allclose(posterior["map_point"], selected)
    assert torch.allclose(
        posterior["map_point"].norm(dim=1),
        torch.ones(3),
        atol=1e-5,
    )
    assert not torch.allclose(
        posterior["map_point"],
        posterior["global_mean"],
    )


def test_candidate_aware_loss_is_finite_and_updates_residual_scorer() -> None:
    model = make_model()
    features = torch.nn.functional.normalize(torch.randn(6, 12), dim=1)
    labels = torch.tensor([0, 1, 2, 3, 4, 0])
    centers = model.centers_unit[labels].clone()
    perturbation = torch.tensor([0.02, -0.01, 0.015]).expand_as(centers)
    perturbation = perturbation - (
        perturbation * centers
    ).sum(dim=1, keepdim=True) * centers
    targets = torch.nn.functional.normalize(centers + perturbation, dim=1)
    loss = model.training_loss(
        features,
        labels,
        targets,
        top_k=3,
        geo_scale_km=50.0,
        near_radius_km=100.0,
        mixture_weight=0.03,
        offset_weight=1.0,
        gate_weight=0.05,
        anchor_weight=0.002,
    )
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert 0.0 <= loss.natural_topk_recall.item() <= 1.0
    assert torch.allclose(
        loss.natural_topk_recall + loss.forced_inclusion_rate,
        torch.tensor(1.0),
    )
    final_layer = model.residual_scorer[-1]
    assert final_layer.weight.grad is not None
    assert torch.isfinite(final_layer.weight.grad).all()
    assert torch.count_nonzero(final_layer.weight.grad) > 0
    assert model.offset_head[-1].weight.grad is not None
    assert torch.count_nonzero(model.offset_head[-1].weight.grad) > 0


def test_density_is_off_by_default_and_explicit_when_enabled() -> None:
    base = make_model()
    features = torch.nn.functional.normalize(torch.randn(2, 12), dim=1)
    assert torch.count_nonzero(
        base.score_components(features)["density_logits"]
    ) == 0

    density_model = PrototypeResidualHeadV2(
        feature_dimension=12,
        prototypes=base.prototypes,
        centers_unit=base.centers_unit,
        leaf_counts=torch.tensor([2, 3, 5, 7, 11]),
        hidden_dimension=8,
        density_weight=0.5,
    )
    density = density_model.score_components(features)["density_logits"]
    assert np.isclose(float(density[0]), -0.5 * np.log(2.0))
