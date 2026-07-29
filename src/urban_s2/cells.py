"""Adaptive, train-only S2 partitions for worldwide geolocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from s2sphere import CellId, LatLng

from .geodesy import latlon_to_unit


@dataclass(frozen=True)
class AdaptiveS2Partition:
    labels: np.ndarray
    leaf_cell_ids: np.ndarray
    leaf_levels: np.ndarray
    leaf_counts: np.ndarray
    leaf_centers_unit: np.ndarray
    leaf_tokens: tuple[str, ...]


@dataclass(frozen=True)
class S2Tree:
    node_cell_ids: np.ndarray
    node_levels: np.ndarray
    node_parents: np.ndarray
    leaf_node_indices: np.ndarray


def _cell_id(latitude: float, longitude: float, level: int) -> int:
    return CellId.from_lat_lng(
        LatLng.from_degrees(float(latitude), float(longitude))
    ).parent(level).id()


def build_adaptive_s2_partition(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    min_level: int = 3,
    max_level: int = 12,
    max_samples_per_leaf: int = 64,
) -> AdaptiveS2Partition:
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if latitude.shape != longitude.shape or latitude.ndim != 1:
        raise ValueError("latitude and longitude must be matching 1D arrays")
    if latitude.size == 0:
        raise ValueError("coordinates must be non-empty")
    if not 0 <= min_level <= max_level <= 30:
        raise ValueError("S2 levels must satisfy 0 <= min <= max <= 30")
    if max_samples_per_leaf < 1:
        raise ValueError("max_samples_per_leaf must be positive")
    if (
        not np.isfinite(latitude).all()
        or not np.isfinite(longitude).all()
        or np.any(np.abs(latitude) > 90)
        or np.any(np.abs(longitude) > 180)
    ):
        raise ValueError("coordinates contain invalid latitude/longitude")

    minimum_cells = np.fromiter(
        (
            _cell_id(lat, lon, min_level)
            for lat, lon in zip(latitude, longitude, strict=True)
        ),
        dtype=np.uint64,
        count=latitude.size,
    )
    leaves: list[tuple[int, int, np.ndarray]] = []

    def split(cell_id: int, level: int, indices: np.ndarray) -> None:
        if len(indices) <= max_samples_per_leaf or level >= max_level:
            leaves.append((cell_id, level, indices))
            return
        next_level = level + 1
        child_ids = np.fromiter(
            (
                _cell_id(latitude[index], longitude[index], next_level)
                for index in indices
            ),
            dtype=np.uint64,
            count=len(indices),
        )
        for child_id in np.unique(child_ids):
            split(
                int(child_id),
                next_level,
                indices[child_ids == child_id],
            )

    all_indices = np.arange(latitude.size, dtype=np.int64)
    for minimum_cell in np.unique(minimum_cells):
        split(
            int(minimum_cell),
            min_level,
            all_indices[minimum_cells == minimum_cell],
        )

    leaves.sort(key=lambda item: item[0])
    labels = np.empty(latitude.size, dtype=np.int64)
    leaf_cell_ids = np.empty(len(leaves), dtype=np.uint64)
    leaf_levels = np.empty(len(leaves), dtype=np.int16)
    leaf_counts = np.empty(len(leaves), dtype=np.int64)
    centers = np.empty((len(leaves), 3), dtype=np.float64)
    tokens: list[str] = []
    coordinates_unit = latlon_to_unit(latitude, longitude)
    for label, (cell_id, level, indices) in enumerate(leaves):
        labels[indices] = label
        leaf_cell_ids[label] = cell_id
        leaf_levels[label] = level
        leaf_counts[label] = len(indices)
        resultant = coordinates_unit[indices].sum(axis=0)
        centers[label] = resultant / np.linalg.norm(resultant)
        tokens.append(CellId(cell_id).to_token())

    return AdaptiveS2Partition(
        labels=labels,
        leaf_cell_ids=leaf_cell_ids,
        leaf_levels=leaf_levels,
        leaf_counts=leaf_counts,
        leaf_centers_unit=centers,
        leaf_tokens=tuple(tokens),
    )


def build_s2_tree(
    leaf_cell_ids: np.ndarray,
    leaf_levels: np.ndarray,
    *,
    min_level: int,
) -> S2Tree:
    cell_ids = np.asarray(leaf_cell_ids, dtype=np.uint64)
    levels = np.asarray(leaf_levels, dtype=np.int16)
    if cell_ids.ndim != 1 or levels.shape != cell_ids.shape:
        raise ValueError("leaf cell IDs and levels must be matching 1D arrays")
    if np.any(levels < min_level):
        raise ValueError("leaf level is below min_level")

    keys: set[tuple[int, int]] = set()
    for cell_id, leaf_level in zip(cell_ids, levels, strict=True):
        cell = CellId(int(cell_id))
        for level in range(min_level, int(leaf_level) + 1):
            ancestor = cell.parent(level)
            keys.add((level, ancestor.id()))
    ordered = sorted(keys)

    node_cell_ids = np.zeros(len(ordered) + 1, dtype=np.uint64)
    node_levels = np.full(len(ordered) + 1, min_level - 1, dtype=np.int16)
    node_parents = np.full(len(ordered) + 1, -1, dtype=np.int64)
    node_by_key: dict[tuple[int, int], int] = {}
    for node_index, (level, cell_id) in enumerate(ordered, start=1):
        node_by_key[(level, cell_id)] = node_index
        node_cell_ids[node_index] = cell_id
        node_levels[node_index] = level
        if level == min_level:
            node_parents[node_index] = 0
        else:
            parent_id = CellId(cell_id).parent(level - 1).id()
            node_parents[node_index] = node_by_key[(level - 1, parent_id)]

    leaf_node_indices = np.fromiter(
        (
            node_by_key[(int(level), int(cell_id))]
            for cell_id, level in zip(cell_ids, levels, strict=True)
        ),
        dtype=np.int64,
        count=len(cell_ids),
    )
    return S2Tree(
        node_cell_ids=node_cell_ids,
        node_levels=node_levels,
        node_parents=node_parents,
        leaf_node_indices=leaf_node_indices,
    )

