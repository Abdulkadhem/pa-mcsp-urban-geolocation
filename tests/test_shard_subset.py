from scripts.extract_osv5m_shard_subset import (
    canonical_shard,
    selection_fingerprint,
)


def test_canonical_train_shard_names() -> None:
    assert canonical_shard("7") == "images/train/07.zip"
    assert canonical_shard("26.zip") == "images/train/26.zip"
    assert (
        canonical_shard("images/train/91.zip")
        == "images/train/91.zip"
    )


def test_selection_fingerprint_is_order_invariant() -> None:
    first = selection_fingerprint(
        "dataset/repo",
        "revision",
        ["images/train/26.zip", "images/train/91.zip"],
    )
    second = selection_fingerprint(
        "dataset/repo",
        "revision",
        ["images/train/91.zip", "images/train/26.zip"],
    )
    assert first == second

