"""Project full-run time and cost from a measured 1% pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PilotMeasurement:
    train_images: int
    test_images: int
    feature_seconds: float
    train_seconds: float
    train_epochs: float
    train_seeds: int
    evaluation_seconds: float
    preprocessing_seconds: float = 0.0


@dataclass(frozen=True)
class RuntimeProjection:
    preprocessing_hours: float
    feature_hours: float
    training_hours: float
    evaluation_hours: float
    total_hours: float
    estimated_gpu_cost_usd: float
    effective_gpu_speedup: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def project_runtime(
    pilot: PilotMeasurement,
    *,
    target_train_images: int,
    target_test_images: int,
    target_epochs: float,
    target_seeds: int,
    gpu_count: int = 1,
    scaling_efficiency: float = 0.85,
    safety_factor: float = 1.25,
    gpu_hourly_price_usd: float = 0.0,
) -> RuntimeProjection:
    """Scale measured phase times while keeping one-time and repeated work separate."""
    if min(
        pilot.train_images,
        pilot.test_images,
        pilot.feature_seconds,
        pilot.train_seconds,
        pilot.train_epochs,
        pilot.train_seeds,
        pilot.evaluation_seconds,
        target_train_images,
        target_test_images,
        target_epochs,
        target_seeds,
        gpu_count,
    ) <= 0:
        raise ValueError("counts, phase times, epochs, seeds, and gpu_count must be positive")
    if not 0 < scaling_efficiency <= 1:
        raise ValueError("scaling_efficiency must lie in (0, 1]")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")
    if gpu_hourly_price_usd < 0:
        raise ValueError("gpu_hourly_price_usd must be non-negative")

    train_scale = target_train_images / pilot.train_images
    test_scale = target_test_images / pilot.test_images
    effective_speedup = 1.0 + (gpu_count - 1) * scaling_efficiency

    preprocessing = pilot.preprocessing_seconds * train_scale
    features = pilot.feature_seconds * train_scale / effective_speedup
    training = (
        pilot.train_seconds
        * train_scale
        * (target_epochs / pilot.train_epochs)
        * (target_seeds / pilot.train_seeds)
        / effective_speedup
    )
    evaluation = pilot.evaluation_seconds * test_scale * target_seeds / effective_speedup

    seconds_per_hour = 3600.0
    phase_hours = [
        safety_factor * phase / seconds_per_hour
        for phase in (preprocessing, features, training, evaluation)
    ]
    total_hours = sum(phase_hours)
    return RuntimeProjection(
        preprocessing_hours=phase_hours[0],
        feature_hours=phase_hours[1],
        training_hours=phase_hours[2],
        evaluation_hours=phase_hours[3],
        total_hours=total_hours,
        estimated_gpu_cost_usd=total_hours * gpu_count * gpu_hourly_price_usd,
        effective_gpu_speedup=effective_speedup,
    )

