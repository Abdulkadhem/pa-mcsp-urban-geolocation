import pytest

from urban_s2.runtime import PilotMeasurement, project_runtime


def pilot():
    return PilotMeasurement(
        train_images=50_000,
        test_images=2_000,
        preprocessing_seconds=3_600,
        feature_seconds=7_200,
        train_seconds=10_800,
        train_epochs=3,
        train_seeds=1,
        evaluation_seconds=1_800,
    )


def test_projection_separates_one_time_features_from_seed_training():
    one_seed = project_runtime(
        pilot(),
        target_train_images=5_000_000,
        target_test_images=200_000,
        target_epochs=3,
        target_seeds=1,
        safety_factor=1.0,
    )
    three_seeds = project_runtime(
        pilot(),
        target_train_images=5_000_000,
        target_test_images=200_000,
        target_epochs=3,
        target_seeds=3,
        safety_factor=1.0,
    )
    assert three_seeds.feature_hours == pytest.approx(one_seed.feature_hours)
    assert three_seeds.training_hours == pytest.approx(3 * one_seed.training_hours)
    assert three_seeds.evaluation_hours == pytest.approx(3 * one_seed.evaluation_hours)


def test_multi_gpu_projection_uses_bounded_efficiency():
    single = project_runtime(
        pilot(),
        target_train_images=5_000_000,
        target_test_images=200_000,
        target_epochs=3,
        target_seeds=3,
        gpu_count=1,
        safety_factor=1.0,
    )
    four = project_runtime(
        pilot(),
        target_train_images=5_000_000,
        target_test_images=200_000,
        target_epochs=3,
        target_seeds=3,
        gpu_count=4,
        scaling_efficiency=0.8,
        safety_factor=1.0,
    )
    assert four.effective_gpu_speedup == pytest.approx(3.4)
    assert four.training_hours == pytest.approx(single.training_hours / 3.4)


def test_cost_accounts_for_gpu_count():
    projected = project_runtime(
        pilot(),
        target_train_images=50_000,
        target_test_images=2_000,
        target_epochs=3,
        target_seeds=1,
        gpu_count=1,
        safety_factor=1.0,
        gpu_hourly_price_usd=0.5,
    )
    assert projected.estimated_gpu_cost_usd == pytest.approx(projected.total_hours * 0.5)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        project_runtime(
            pilot(),
            target_train_images=5_000_000,
            target_test_images=200_000,
            target_epochs=3,
            target_seeds=3,
            gpu_count=0,
        )

