import numpy as np
import pandas as pd
import pytest

from urban_s2.spatial_split import (
    SpatialSplitConfig,
    build_spatial_feature_index,
    rows_within_geodesic_buffer,
)


def _row(
    image_id,
    sequence,
    latitude,
    longitude,
    country,
    official_split="train",
    split="train",
):
    return {
        "feature_row": int(image_id),
        "id": str(image_id),
        "sequence": sequence,
        "latitude": latitude,
        "longitude": longitude,
        "country": country,
        "official_split": official_split,
        "split": split,
    }


def _representative_index():
    rows = [
        _row(1, "iq-near-a", 33.300000, 44.300000, "IQ"),
        _row(2, "iq-near-a", 33.300010, 44.300010, "IQ"),
        _row(3, "iq-near-b", 33.300000, 44.306000, "IQ"),
        _row(4, "iq-far", 33.500000, 44.500000, "IQ"),
        _row(5, "us-a", 40.700000, -74.000000, "US"),
        _row(6, "us-b", 40.900000, -73.700000, "US"),
        _row(
            7,
            "official-test",
            33.300000,
            44.300000,
            "IQ",
            official_split="test",
            split="official_test_pilot",
        ),
    ]
    return pd.DataFrame(rows)


def test_geodesic_buffer_is_exact_across_dateline():
    marked = rows_within_geodesic_buffer(
        np.array([0.0, 0.0]),
        np.array([179.999, 179.0]),
        np.array([0.0]),
        np.array([-179.999]),
        buffer_km=1.0,
    )
    assert marked.tolist() == [True, False]


def test_spatial_split_is_sequence_safe_buffered_and_test_isolated():
    source = _representative_index()
    config = SpatialSplitConfig(
        development_fraction=0.5,
        seed=17,
        block_size_km=0.1,
        buffer_km=0.08,
    )
    output, stats = build_spatial_feature_index(source, config)

    assert len(output) == len(source)
    assert output["feature_row"].tolist() == source["feature_row"].tolist()
    test_row = output.loc[output["official_split"].eq("test")].iloc[0]
    assert test_row["split"] == "official_test_pilot"
    assert test_row["spatial_block_id"] is None

    per_sequence_splits = (
        output.loc[output["official_split"].eq("train")]
        .groupby("sequence")["split"]
        .nunique()
    )
    assert per_sequence_splits.max() == 1
    assert stats["leakage_audit"]["train_development_sequence_overlap"] == 0
    assert stats["leakage_audit"]["train_development_block_overlap"] == 0
    assert stats["leakage_audit"]["violating_train_rows_after_exclusion"] == 0
    assert stats["leakage_audit"]["geodesic_buffer_verified"] is True
    assert stats["leakage_audit"]["official_test_used_in_split_construction"] is False


def test_split_is_reproducible_and_input_order_invariant():
    source = _representative_index()
    config = SpatialSplitConfig(
        development_fraction=0.4,
        seed=20260725,
        block_size_km=5.0,
        buffer_km=1.0,
    )
    first, first_stats = build_spatial_feature_index(source, config)
    shuffled = source.sample(frac=1.0, random_state=999).reset_index(drop=True)
    second, second_stats = build_spatial_feature_index(shuffled, config)

    first_assignment = first.set_index("id")["split"].sort_index()
    second_assignment = second.set_index("id")["split"].sort_index()
    pd.testing.assert_series_equal(first_assignment, second_assignment)
    assert first_stats["country_stats"] == second_stats["country_stats"]
    assert first_stats["leakage_audit"] == second_stats["leakage_audit"]


def test_country_prefix_is_part_of_every_train_block():
    output, _ = build_spatial_feature_index(
        _representative_index(),
        SpatialSplitConfig(
            development_fraction=0.5,
            seed=3,
            block_size_km=5.0,
            buffer_km=1.0,
        ),
    )
    train_source = output.loc[output["official_split"].eq("train")]
    assert all(
        block.startswith(f"{country}:")
        for block, country in zip(
            train_source["spatial_block_id"],
            train_source["country"],
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "config",
    [
        SpatialSplitConfig(development_fraction=0.0),
        SpatialSplitConfig(development_fraction=1.0),
        SpatialSplitConfig(block_size_km=0.5, buffer_km=1.0),
    ],
)
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        build_spatial_feature_index(_representative_index(), config)


def test_invalid_official_coordinates_are_rejected():
    source = _representative_index()
    source.loc[0, "latitude"] = 91.0
    with pytest.raises(ValueError, match="latitude"):
        build_spatial_feature_index(source, SpatialSplitConfig())
