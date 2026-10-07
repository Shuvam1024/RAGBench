import math

import pytest

from ragstat.evaluation.retrieval import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.mark.parametrize(("k", "expected"), [(1, 0), (2, 0.5), (3, 1), (100, 1)])
def test_recall(k: int, expected: float) -> None:
    assert recall_at_k(["B", "B", "C", "A"], {"A", "C"}, k) == expected


def test_reciprocal_rank_and_mean() -> None:
    assert reciprocal_rank(["B", "B", "C", "A"], {"A", "C"}) == 0.5
    assert reciprocal_rank(["B"], {"A"}) == 0
    assert reciprocal_rank([], {"A"}) == 0
    assert recall_at_k([], {"A"}, 1) == 0
    assert mean_reciprocal_rank([["B", "C", "A"], ["A"]], [{"A", "C"}, {"A"}]) == 0.75


def test_precision_average_precision_and_ndcg() -> None:
    ranking = ["B", "C", "A"]
    relevant = {"A", "C"}
    assert precision_at_k(ranking, relevant, 1) == 0
    assert precision_at_k(ranking, relevant, 2) == 0.5
    assert precision_at_k(ranking, relevant, 3) == pytest.approx(2 / 3)
    assert average_precision(ranking, relevant) == pytest.approx((0.5 + 2 / 3) / 2)
    assert average_precision(["B"], relevant) == 0

    def dcg(grades: list[float]) -> float:
        return sum(
            (2.0**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1)
        )

    binary = ndcg_at_k(ranking, {"A": 1, "C": 1}, 3)
    assert binary == pytest.approx(dcg([0, 1, 1]) / dcg([1, 1]))
    graded = ndcg_at_k(["A", "B", "C"], {"A": 3, "C": 1}, 3)
    assert graded == pytest.approx(dcg([3, 0, 1]) / dcg([3, 1]))
    assert ndcg_at_k(["A"], {"A": 2}, 5) == 1


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
            recall_at_k(["a"], ["a"], k)
        with pytest.raises(ValueError):
            precision_at_k(["a"], ["a"], k)
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], {"a": 1}, k)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {}, 1)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 0}, 1)
    with pytest.raises(ValueError):
        average_precision(["a"], [])
