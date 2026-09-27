import pytest

from ragbench.evaluation.retrieval import mean_reciprocal_rank, recall_at_k, reciprocal_rank


@pytest.mark.parametrize(("k", "expected"), [(1, 0), (2, 0.5), (3, 1), (100, 1)])
def test_recall(k: int, expected: float) -> None:
    assert recall_at_k(["B", "B", "C", "A"], {"A", "C"}, k) == expected


def test_reciprocal_rank_and_mean() -> None:
    assert reciprocal_rank(["B", "B", "C", "A"], {"A", "C"}) == 0.5
    assert reciprocal_rank(["B"], {"A"}) == 0
    assert reciprocal_rank([], {"A"}) == 0
    assert recall_at_k([], {"A"}, 1) == 0
    assert mean_reciprocal_rank([["B", "C", "A"], ["A"]], [{"A", "C"}, {"A"}]) == 0.75


def test_invalid_metric_inputs() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], [], 1)
    with pytest.raises(ValueError):
        reciprocal_rank([], [])
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [])
    for k in (0, -1, True):
        with pytest.raises(ValueError):
            recall_at_k([], ["a"], k)
