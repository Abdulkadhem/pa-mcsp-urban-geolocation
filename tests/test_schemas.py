from urban_s2.schemas import validate_manifest_row, validate_prediction_row


def valid_manifest():
    return {
        "source_dataset": "osv5m",
        "source_revision": "locked-revision",
        "image_id": "image-1",
        "relative_path": "train/000/image-1.jpg",
        "official_split": "train",
        "development_split": "validation",
        "latitude": 33.3152,
        "longitude": 44.3661,
        "sequence_id": "sequence-1",
        "ghsl_smod_class": 30,
        "urban_flag": True,
        "decode_ok": True,
        "sha256": "a" * 64,
        "license_spdx": "CC-BY-SA-4.0",
    }


def valid_prediction():
    return {
        "run_id": "run-1",
        "query_id": "query-1",
        "method": "urban-s2-posterior",
        "seed": 20260724,
        "official_split": "validation",
        "urban_flag": True,
        "ground_truth_latitude": 33.3152,
        "ground_truth_longitude": 44.3661,
        "predicted_latitude": 33.30,
        "predicted_longitude": 44.35,
        "error_km": 2.3,
        "confidence": 0.72,
        "latency_ms": 18.4,
        "failure_code": "",
    }


def test_valid_rows_pass():
    assert validate_manifest_row(valid_manifest()) == []
    assert validate_prediction_row(valid_prediction()) == []


def test_manifest_rejects_invalid_coordinates_and_hash():
    row = valid_manifest()
    row["latitude"] = 91
    row["sha256"] = "placeholder"
    errors = validate_manifest_row(row)
    assert "latitude/longitude are invalid" in errors
    assert "sha256 must be 64 lowercase hexadecimal characters" in errors


def test_prediction_requires_failure_code_for_invalid_coordinate():
    row = valid_prediction()
    row["predicted_latitude"] = None
    row["predicted_longitude"] = None
    assert "successful predictions require a valid coordinate" in validate_prediction_row(row)
    row["failure_code"] = "timeout"
    assert "successful predictions require a valid coordinate" not in validate_prediction_row(row)


def test_prediction_rejects_out_of_range_confidence():
    row = valid_prediction()
    row["confidence"] = 1.5
    assert "confidence must lie in [0, 1]" in validate_prediction_row(row)

