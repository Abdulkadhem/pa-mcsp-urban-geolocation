import json
from pathlib import Path

from scripts.verify_paper_release import validate_config


def test_locked_paper_release_manifest_is_consistent():
    root = Path(__file__).parents[1]
    payload = validate_config(root / "configs" / "pa_mcsp_paper_locked.json")
    assert payload["official_queries"] == 38190
    assert payload["official_metrics"]["accuracy_25km"] == 0.2023042681330191


def test_locked_manifest_rejects_trainable_checkpoint(tmp_path):
    root = Path(__file__).parents[1]
    payload = json.loads((root / "configs" / "pa_mcsp_paper_locked.json").read_text())
    payload["trainable_checkpoint"] = "weights/example.pt"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    try:
        validate_config(path)
    except ValueError as exc:
        assert "learned-free" in str(exc)
    else:
        raise AssertionError("trainable checkpoint must be rejected")
