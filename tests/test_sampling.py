import pytest

from urban_s2.sampling import stable_group_fraction, stable_group_selected


def test_sampling_is_deterministic_and_seeded():
    first = stable_group_fraction("sequence-17", 20260724)
    assert first == stable_group_fraction("sequence-17", 20260724)
    assert first != stable_group_fraction("sequence-17", 20260725)


def test_expected_one_percent_rate_is_close_over_many_groups():
    selected = sum(
        stable_group_selected(f"sequence-{index}", 0.01, 20260724)
        for index in range(100_000)
    )
    assert 800 <= selected <= 1_200


def test_whole_group_receives_one_decision():
    decisions = {
        stable_group_selected("shared-sequence", 0.01, 20260724)
        for _ in range(100)
    }
    assert len(decisions) == 1


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        stable_group_selected("sequence", fraction, 1)

