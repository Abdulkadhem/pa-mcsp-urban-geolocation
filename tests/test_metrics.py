import numpy as np
import pytest

from urban_s2.metrics import evaluate_point_predictions


def test_exact_predictions_have_perfect_accuracy():
    coordinates = np.array([[0.0, 0.0], [33.3152, 44.3661]])
    result = evaluate_point_predictions(coordinates, coordinates.copy())
    assert result["n_failures"] == 0
    assert result["accuracy"]["1"] == pytest.approx(1.0)
    assert result["mean_error_km"] == pytest.approx(0.0)


def test_failures_remain_in_denominator_and_count_as_incorrect():
    ground_truth = np.array([[0.0, 0.0], [0.0, 0.0]])
    predictions = np.array([[0.0, 0.0], [np.nan, np.nan]])
    result = evaluate_point_predictions(ground_truth, predictions)
    assert result["n_queries"] == 2
    assert result["n_failures"] == 1
    assert result["failure_rate"] == pytest.approx(0.5)
    assert result["accuracy"]["2500"] == pytest.approx(0.5)


def test_declared_failure_overrides_valid_coordinate():
    ground_truth = np.array([[0.0, 0.0]])
    predictions = np.array([[0.0, 0.0]])
    result = evaluate_point_predictions(
        ground_truth, predictions, failure_mask=np.array([True])
    )
    assert result["accuracy"]["2500"] == pytest.approx(0.0)


def test_invalid_ground_truth_is_rejected():
    with pytest.raises(ValueError):
        evaluate_point_predictions(
            np.array([[95.0, 0.0]]),
            np.array([[0.0, 0.0]]),
        )

