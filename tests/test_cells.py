from __future__ import annotations

import numpy as np

from urban_s2.cells import build_adaptive_s2_partition, build_s2_tree


def test_adaptive_partition_is_complete_and_bounded() -> None:
    latitude = np.concatenate(
        (np.linspace(40.61, 40.81, 20), np.array([48.85] * 3 + [-33.87] * 4))
    )
    longitude = np.concatenate(
        (np.linspace(-74.1, -73.9, 20), np.array([2.35] * 3 + [151.21] * 4))
    )
    partition = build_adaptive_s2_partition(
        latitude,
        longitude,
        min_level=3,
        max_level=12,
        max_samples_per_leaf=8,
    )

    assert partition.labels.shape == latitude.shape
    assert np.all(partition.labels >= 0)
    assert partition.leaf_counts.sum() == latitude.size
    assert partition.leaf_counts.max() <= 8
    assert np.allclose(
        np.linalg.norm(partition.leaf_centers_unit, axis=1),
        1.0,
    )


def test_s2_tree_has_valid_parent_order_and_leaf_mapping() -> None:
    latitude = np.array([35.0, 35.01, 51.5, -23.5])
    longitude = np.array([139.0, 139.01, -0.1, -46.6])
    partition = build_adaptive_s2_partition(
        latitude,
        longitude,
        min_level=2,
        max_level=8,
        max_samples_per_leaf=1,
    )
    tree = build_s2_tree(
        partition.leaf_cell_ids,
        partition.leaf_levels,
        min_level=2,
    )

    assert tree.node_parents[0] == -1
    assert np.all(tree.node_parents[1:] < np.arange(1, len(tree.node_parents)))
    assert len(np.unique(tree.leaf_node_indices)) == len(
        partition.leaf_cell_ids
    )
