from pathlib import Path

import numpy as np
import pandas as pd

from scripts.seed_expanded_feature_run import sha256_file


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"urban-s2")
    assert sha256_file(target) == (
        "296114928a28c85d4b894d86b3bea1fa"
        "fc96a5d7785f8497258cbba6f7cb54ea"
    )


def test_old_id_rank_preserves_feature_order() -> None:
    old_ids = pd.Series(["b", "a"])
    expanded = pd.DataFrame({"id": ["a", "b", "c"]})
    rank = pd.Series(np.arange(len(old_ids)), index=old_ids.to_numpy())
    expanded["_old_rank"] = expanded["id"].map(rank)
    reused = expanded.loc[expanded["_old_rank"].notna()].sort_values("_old_rank")
    assert reused["id"].tolist() == ["b", "a"]
