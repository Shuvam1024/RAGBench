import pytest

from ragstat.evaluation.statistics import bootstrap_mean_ci, signflip_p_value


def test_bootstrap_is_seeded_and_constant_intervals_collapse() -> None:
    kwargs = {"samples": 200, "seed": "fixed", "confidence": 0.9}
    first = bootstrap_mean_ci([0.1, 0.4, 0.0, 1.0], **kwargs)
    assert first == bootstrap_mean_ci([0.1, 0.4, 0.0, 1.0], **kwargs)
    assert first != bootstrap_mean_ci(
        [0.1, 0.4, 0.0, 1.0], samples=200, seed="other", confidence=0.9
    )
    mean, low, high = first
    assert low <= mean <= high
    assert bootstrap_mean_ci([0.2, 0.2], samples=20, seed="c", confidence=0.95) == (0.2, 0.2, 0.2)


def test_signflip_p_values() -> None:
    assert signflip_p_value([0.0, 0.0, 0.0], samples=50, seed="zero") == 1
    assert signflip_p_value([1, 1, 1, 1, 1, 1, 1, 1], samples=2000, seed="shift") < 0.05
    assert signflip_p_value([1, -1], samples=100, seed="a") == signflip_p_value(
        [1, -1], samples=100, seed="a"
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: bootstrap_mean_ci([], samples=1, seed="x", confidence=0.9),
        lambda: bootstrap_mean_ci([1.0], samples=0, seed="x", confidence=0.9),
        lambda: bootstrap_mean_ci([1.0], samples=True, seed="x", confidence=0.9),
        lambda: bootstrap_mean_ci([1.0], samples=1, seed="x", confidence=0),
        lambda: signflip_p_value([], samples=1, seed="x"),
        lambda: signflip_p_value([1.0], samples=-1, seed="x"),
    ],
)
def test_statistics_reject_invalid_inputs(call: object) -> None:
    with pytest.raises(ValueError):
        call()
