"""Leakage-resistant, reproducible spatial splits for urban geolocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from urban_s2.geodesy import EARTH_MEAN_RADIUS_KM, latlon_to_unit
from urban_s2.sampling import stable_group_fraction


@dataclass(frozen=True)
class SpatialSplitConfig:
    """Configuration for a country-stratified, sequence-safe spatial split."""

    development_fraction: float = 0.1
    seed: int = 20260725
    block_size_km: float = 25.0
    buffer_km: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.development_fraction < 1.0:
            raise ValueError("development_fraction must lie in (0, 1)")
        if self.block_size_km <= 0.0:
            raise ValueError("block_size_km must be positive")
        if self.buffer_km <= 0.0:
            raise ValueError("buffer_km must be positive")
        if self.buffer_km >= self.block_size_km:
            raise ValueError("buffer_km must be smaller than block_size_km")


def _validate_feature_index(frame: Any) -> None:
    required = {
        "id",
        "split",
        "official_split",
        "sequence",
        "latitude",
        "longitude",
        "country",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature index is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("feature index is empty")
    if frame["id"].isna().any() or frame["id"].astype(str).duplicated().any():
        raise ValueError("feature index IDs must be non-null and unique")

    official_splits = set(frame["official_split"].dropna().astype(str).unique())
    unexpected = official_splits - {"train", "test"}
    if unexpected:
        raise ValueError(f"unexpected official_split values: {sorted(unexpected)}")

    train = frame.loc[frame["official_split"].astype(str).eq("train")]
    if train.empty:
        raise ValueError("feature index has no official train rows")
    if train["sequence"].isna().any() or train["sequence"].astype(str).eq("").any():
        raise ValueError("official train rows require non-empty sequence IDs")

    latitude = train["latitude"].to_numpy(dtype=np.float64)
    longitude = train["longitude"].to_numpy(dtype=np.float64)
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("official train coordinates must be finite")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("latitude must lie in [-90, 90]")
    if np.any((longitude < -180.0) | (longitude > 180.0)):
        raise ValueError("longitude must lie in [-180, 180]")


def _sequence_table(train: Any, config: SpatialSplitConfig) -> Any:
    """Return one deterministic representative and country per sequence."""
    import pandas as pd

    working = train[
        ["id", "sequence", "latitude", "longitude", "country"]
    ].copy()
    working["id"] = working["id"].astype(str)
    working["sequence"] = working["sequence"].astype(str)
    working["country"] = (
        working["country"].astype("string").fillna("UNKNOWN").replace("", "UNKNOWN")
    )
    working = working.sort_values(["sequence", "id"], kind="mergesort")

    units = latlon_to_unit(
        working["latitude"].to_numpy(dtype=np.float64),
        working["longitude"].to_numpy(dtype=np.float64),
    )
    working["_x"] = units[:, 0]
    working["_y"] = units[:, 1]
    working["_z"] = units[:, 2]

    grouped = working.groupby("sequence", sort=True, observed=True)
    representatives = grouped.agg(
        row_count=("id", "size"),
        x=("_x", "mean"),
        y=("_y", "mean"),
        z=("_z", "mean"),
    ).reset_index()
    representative_units = representatives[["x", "y", "z"]].to_numpy(
        dtype=np.float64
    )
    norms = np.linalg.norm(representative_units, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("a sequence has an undefined spherical representative")
    representative_units /= norms
    representatives[["x", "y", "z"]] = representative_units

    country_counts = (
        working.groupby(["sequence", "country"], sort=True, observed=True)
        .size()
        .rename("country_rows")
        .reset_index()
    )
    country_counts = country_counts.sort_values(
        ["sequence", "country_rows", "country"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    modal_country = country_counts.drop_duplicates(
        "sequence", keep="first"
    )[["sequence", "country"]]
    representatives = representatives.merge(
        modal_country, on="sequence", how="left", validate="one_to_one"
    )

    # Quantising unit-sphere Cartesian coordinates avoids dateline and polar
    # discontinuities. The country prefix makes block selection stratified.
    cell_width = config.block_size_km / EARTH_MEAN_RADIUS_KM
    indices = np.floor(
        (representatives[["x", "y", "z"]].to_numpy() + 1.0) / cell_width
    ).astype(np.int64)
    representatives["spatial_block_id"] = [
        f"{country}:{ix}:{iy}:{iz}"
        for country, (ix, iy, iz) in zip(
            representatives["country"].astype(str), indices, strict=True
        )
    ]
    return representatives


def _select_development_blocks(
    sequences: Any, config: SpatialSplitConfig
) -> set[str]:
    """Select whole blocks independently within each country."""
    blocks = (
        sequences.groupby(
            ["country", "spatial_block_id"], sort=True, observed=True
        )
        .agg(row_count=("row_count", "sum"), sequence_count=("sequence", "size"))
        .reset_index()
    )
    selected: set[str] = set()
    for country, country_blocks in blocks.groupby(
        "country", sort=True, observed=True
    ):
        country_blocks = country_blocks.copy()
        if len(country_blocks) < 2:
            # Never place a country's sole block entirely in development.
            continue
        country_blocks["_rank"] = country_blocks["spatial_block_id"].map(
            lambda block: stable_group_fraction(
                f"{country}:{block}", config.seed
            )
        )
        country_blocks = country_blocks.sort_values(
            ["_rank", "spatial_block_id"], kind="mergesort"
        )
        cumulative = country_blocks["row_count"].cumsum().to_numpy(dtype=np.int64)
        target = (
            float(country_blocks["row_count"].sum())
            * config.development_fraction
        )
        # k is constrained to [1, n_blocks - 1], preserving both splits.
        candidate_k = np.arange(1, len(country_blocks), dtype=np.int64)
        best_k = int(
            candidate_k[
                np.argmin(np.abs(cumulative[candidate_k - 1] - target))
            ]
        )
        selected.update(
            country_blocks.iloc[:best_k]["spatial_block_id"].astype(str)
        )
    return selected


def rows_within_geodesic_buffer(
    query_latitude: np.ndarray,
    query_longitude: np.ndarray,
    reference_latitude: np.ndarray,
    reference_longitude: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    """Mark query rows no farther than ``buffer_km`` from any reference.

    A Cartesian hash on unit-sphere coordinates limits exact chord comparisons
    to 27 neighbouring buckets. Chord distance is monotonic with great-circle
    distance, so this is an exact threshold test rather than a projection
    approximation.
    """
    if buffer_km <= 0.0:
        raise ValueError("buffer_km must be positive")
    query = latlon_to_unit(query_latitude, query_longitude).reshape(-1, 3)
    reference = latlon_to_unit(
        reference_latitude, reference_longitude
    ).reshape(-1, 3)
    result = np.zeros(len(query), dtype=bool)
    if len(query) == 0 or len(reference) == 0:
        return result

    angular_radius = buffer_km / EARTH_MEAN_RADIUS_KM
    chord_radius = 2.0 * np.sin(angular_radius / 2.0)
    cell_width = chord_radius
    reference_cells = np.floor((reference + 1.0) / cell_width).astype(np.int64)
    query_cells = np.floor((query + 1.0) / cell_width).astype(np.int64)

    reference_buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(reference_cells):
        reference_buckets.setdefault(tuple(cell), []).append(index)
    query_buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(query_cells):
        query_buckets.setdefault(tuple(cell), []).append(index)

    offsets = tuple(product((-1, 0, 1), repeat=3))
    threshold_squared = chord_radius * chord_radius
    for cell, query_indices_list in query_buckets.items():
        candidate_lists = [
            reference_buckets[
                (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            ]
            for dx, dy, dz in offsets
            if (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            in reference_buckets
        ]
        if not candidate_lists:
            continue
        reference_indices = np.fromiter(
            (item for items in candidate_lists for item in items),
            dtype=np.int64,
        )
        query_indices = np.asarray(query_indices_list, dtype=np.int64)
        candidate_reference = reference[reference_indices]
        # Bound memory in unusually dense urban buckets.
        for start in range(0, len(query_indices), 2048):
            batch_indices = query_indices[start : start + 2048]
            differences = (
                query[batch_indices, None, :] - candidate_reference[None, :, :]
            )
            squared = np.einsum(
                "ijk,ijk->ij", differences, differences, optimize=True
            )
            result[batch_indices] = np.any(
                squared <= threshold_squared, axis=1
            )
    return result


def _split_counts(frame: Any) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split, rows in frame.groupby("split", sort=True, observed=True):
        counts[str(split)] = {
            "rows": int(len(rows)),
            "sequences": int(rows["sequence"].astype(str).nunique()),
            "countries": int(
                rows["country"]
                .astype("string")
                .fillna("UNKNOWN")
                .replace("", "UNKNOWN")
                .nunique()
            ),
        }
    return counts


def build_spatial_feature_index(
    feature_index: Any, config: SpatialSplitConfig
) -> tuple[Any, dict[str, Any]]:
    """Build a spatial split while preserving feature-row alignment.

    Official test rows are retained unchanged as ``official_test_pilot`` for
    array alignment, but are never used for block selection or buffer checks.
    """
    config.validate()
    _validate_feature_index(feature_index)
    output = feature_index.copy()
    output["source_split"] = output["split"].astype(str)
    output["spatial_block_id"] = None
    output["buffer_excluded"] = False

    official_train_mask = output["official_split"].astype(str).eq("train")
    official_test_mask = output["official_split"].astype(str).eq("test")
    train_source = output.loc[official_train_mask].copy()
    sequences = _sequence_table(train_source, config)
    development_blocks = _select_development_blocks(sequences, config)
    if not development_blocks:
        raise ValueError(
            "no development blocks were selected; provide more spatial blocks"
        )

    sequence_to_block = sequences.set_index("sequence")[
        "spatial_block_id"
    ].to_dict()
    train_blocks = (
        output.loc[official_train_mask, "sequence"]
        .astype(str)
        .map(sequence_to_block)
    )
    if train_blocks.isna().any():
        raise RuntimeError("not every official train sequence received a block")
    output.loc[official_train_mask, "spatial_block_id"] = train_blocks
    initial_development_mask = (
        official_train_mask
        & output["spatial_block_id"].astype(str).isin(development_blocks)
    )
    initial_train_mask = official_train_mask & ~initial_development_mask
    output.loc[initial_train_mask, "split"] = "train"
    output.loc[initial_development_mask, "split"] = "development"

    near_development = rows_within_geodesic_buffer(
        output.loc[initial_train_mask, "latitude"].to_numpy(dtype=np.float64),
        output.loc[initial_train_mask, "longitude"].to_numpy(dtype=np.float64),
        output.loc[initial_development_mask, "latitude"].to_numpy(
            dtype=np.float64
        ),
        output.loc[initial_development_mask, "longitude"].to_numpy(
            dtype=np.float64
        ),
        config.buffer_km,
    )
    violating_train_rows = output.loc[initial_train_mask].iloc[
        np.flatnonzero(near_development)
    ]
    excluded_sequences = set(
        violating_train_rows["sequence"].astype(str).unique()
    )
    excluded_mask = (
        official_train_mask
        & output["sequence"].astype(str).isin(excluded_sequences)
    )
    output.loc[excluded_mask, "split"] = "buffer_excluded"
    output.loc[excluded_mask, "buffer_excluded"] = True

    # Test rows remain aligned with their features but are explicitly
    # quarantined from all split construction.
    output.loc[official_test_mask, "split"] = "official_test_pilot"
    output.loc[official_test_mask, "spatial_block_id"] = None
    output.loc[official_test_mask, "buffer_excluded"] = False

    final_train_mask = output["split"].astype(str).eq("train")
    development_mask = output["split"].astype(str).eq("development")
    if not final_train_mask.any() or not development_mask.any():
        raise ValueError("spatial splitting produced an empty train or development set")

    violations_after = rows_within_geodesic_buffer(
        output.loc[final_train_mask, "latitude"].to_numpy(dtype=np.float64),
        output.loc[final_train_mask, "longitude"].to_numpy(dtype=np.float64),
        output.loc[development_mask, "latitude"].to_numpy(dtype=np.float64),
        output.loc[development_mask, "longitude"].to_numpy(dtype=np.float64),
        config.buffer_km,
    )
    train_sequences = set(
        output.loc[final_train_mask, "sequence"].astype(str)
    )
    development_sequences = set(
        output.loc[development_mask, "sequence"].astype(str)
    )
    train_blocks_final = set(
        output.loc[final_train_mask, "spatial_block_id"].astype(str)
    )
    development_blocks_final = set(
        output.loc[development_mask, "spatial_block_id"].astype(str)
    )

    country_stats: dict[str, dict[str, int]] = {}
    official_train = output.loc[official_train_mask]
    normalized_country = (
        official_train["country"]
        .astype("string")
        .fillna("UNKNOWN")
        .replace("", "UNKNOWN")
    )
    for country in sorted(normalized_country.astype(str).unique()):
        country_rows = official_train.loc[
            normalized_country.astype(str).eq(country)
        ]
        country_stats[country] = {
            "official_train_rows": int(len(country_rows)),
            "official_train_sequences": int(
                country_rows["sequence"].astype(str).nunique()
            ),
            "spatial_blocks": int(
                country_rows["spatial_block_id"].astype(str).nunique()
            ),
            "train_rows": int(country_rows["split"].astype(str).eq("train").sum()),
            "development_rows": int(
                country_rows["split"].astype(str).eq("development").sum()
            ),
            "buffer_excluded_rows": int(
                country_rows["split"].astype(str).eq("buffer_excluded").sum()
            ),
        }

    stats: dict[str, Any] = {
        "schema_version": 1,
        "method": "country_stratified_unit_sphere_blocks_with_geodesic_buffer",
        "config": asdict(config),
        "input": {
            "rows": int(len(feature_index)),
            "official_train_rows": int(official_train_mask.sum()),
            "official_test_rows_isolated": int(official_test_mask.sum()),
        },
        "output": {
            "rows": int(len(output)),
            "split_counts": _split_counts(output),
        },
        "leakage_audit": {
            "train_development_id_overlap": int(
                len(
                    set(output.loc[final_train_mask, "id"].astype(str))
                    & set(output.loc[development_mask, "id"].astype(str))
                )
            ),
            "train_development_sequence_overlap": int(
                len(train_sequences & development_sequences)
            ),
            "train_development_block_overlap": int(
                len(train_blocks_final & development_blocks_final)
            ),
            "buffer_km": float(config.buffer_km),
            "violating_train_rows_before_exclusion": int(
                len(violating_train_rows)
            ),
            "buffer_excluded_sequences": int(len(excluded_sequences)),
            "buffer_excluded_rows": int(excluded_mask.sum()),
            "violating_train_rows_after_exclusion": int(
                violations_after.sum()
            ),
            "minimum_train_development_distance_lower_bound_km": float(
                config.buffer_km
            ),
            "geodesic_buffer_verified": bool(not violations_after.any()),
            "official_test_used_in_split_construction": False,
        },
        "country_stats": country_stats,
    }
    if (
        stats["leakage_audit"]["train_development_sequence_overlap"] != 0
        or stats["leakage_audit"]["train_development_block_overlap"] != 0
        or stats["leakage_audit"]["violating_train_rows_after_exclusion"] != 0
    ):
        raise RuntimeError("spatial split leakage audit failed")
    return output, stats
