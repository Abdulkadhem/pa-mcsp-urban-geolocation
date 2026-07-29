from __future__ import annotations

import numpy as np
import torch

from scripts.train_urban_s2_posterior import UrbanS2Posterior
from urban_s2.cells import build_adaptive_s2_partition, build_s2_tree


def test_hierarchical_posterior_is_normalized_and_differentiable() -> None:
    latitude = np.array([40.7, 40.8, 48.8, 51.5, -33.8, -23.5])
    longitude = np.array([-74.0, -73.9, 2.3, -0.1, 151.2, -46.6])
    partition = build_adaptive_s2_partition(
        latitude,
        longitude,
        min_level=2,
        max_level=8,
        max_samples_per_leaf=2,
    )
    tree = build_s2_tree(
        partition.leaf_cell_ids,
        partition.leaf_levels,
        min_level=2,
    )
    generator = np.random.default_rng(7)
    prototypes = generator.normal(
        size=(len(partition.leaf_counts), 16)
    ).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    model = UrbanS2Posterior(
        feature_dimension=16,
        tree=tree,
        prototypes=prototypes,
        centers_unit=partition.leaf_centers_unit,
        leaf_counts=partition.leaf_counts,
    )
    features = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)

    log_probabilities, router_log_probabilities = (
        model.fused_log_probabilities(features)
    )
    posterior = model.posterior(features, top_k=3)
    loss = -log_probabilities[:, 0].mean()
    loss.backward()

    assert torch.allclose(
        torch.logsumexp(log_probabilities, dim=1),
        torch.zeros(4),
        atol=1e-5,
    )
    assert torch.allclose(
        torch.logsumexp(router_log_probabilities, dim=1),
        torch.zeros(4),
        atol=1e-5,
    )
    assert torch.allclose(
        posterior["point"].norm(dim=1),
        torch.ones(4),
        atol=1e-5,
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
