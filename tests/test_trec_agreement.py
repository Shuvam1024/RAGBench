"""Cross-check retrieval metrics against pytrec_eval on a fixed fixture.

pytrec_eval's ``ndcg_cut`` uses the relevance grade itself as the gain.
RAGBench uses exponential gain ``2**grade - 1``. Those definitions agree on
binary labels, which is the SciFact case, and differ when a grade exceeds 1.
"""

from statistics import fmean

import pytest
import pytrec_eval

from ragbench.evaluation.retrieval import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

_RANKINGS = (
    ["b", "a", "c"],
    ["a", "b"],
    ["c", "b", "a"],
)
_GRADES = (
    {"a": 1, "c": 1},
    {"a": 1},
    {"a": 1, "b": 1},
)
_MEASURES = {"map", "recip_rank", "ndcg_cut_10", "recall_10", "P_10"}


def _qrels_and_run(
    rankings: tuple[list[str], ...], grades: tuple[dict[str, int], ...]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for index, (ranking, grade_map) in enumerate(zip(rankings, grades, strict=True)):
        query_id = f"q{index}"
        qrels[query_id] = dict(grade_map)
        # Higher scores rank first. Scores are unique so tie-breaking cannot diverge.
        run[query_id] = {doc_id: float(len(ranking) - rank) for rank, doc_id in enumerate(ranking)}
    return qrels, run


def _mean(measure: str, evaluated: dict[str, dict[str, float]]) -> float:
    return fmean(row[measure] for row in evaluated.values())


def test_binary_metrics_match_pytrec_eval() -> None:
    qrels, run = _qrels_and_run(_RANKINGS, _GRADES)
    evaluated = pytrec_eval.RelevanceEvaluator(qrels, _MEASURES).evaluate(run)
    pairs = list(zip(_RANKINGS, _GRADES, strict=True))
    assert _mean("map", evaluated) == pytest.approx(
        fmean(average_precision(ranking, grades) for ranking, grades in pairs)
    )
    assert _mean("recip_rank", evaluated) == pytest.approx(
        fmean(reciprocal_rank(ranking, grades) for ranking, grades in pairs)
    )
    assert _mean("ndcg_cut_10", evaluated) == pytest.approx(
        fmean(ndcg_at_k(ranking, grades, 10) for ranking, grades in pairs)
    )
    assert _mean("recall_10", evaluated) == pytest.approx(
        fmean(recall_at_k(ranking, grades, 10) for ranking, grades in pairs)
    )
    assert _mean("P_10", evaluated) == pytest.approx(
        fmean(precision_at_k(ranking, grades, 10) for ranking, grades in pairs)
    )


def test_graded_labels_match_pytrec_eval_except_exponential_ndcg() -> None:
    ranking = ["b", "a", "c"]
    grades = {"a": 1, "c": 2}
    qrels, run = _qrels_and_run((ranking,), (grades,))
    evaluated = pytrec_eval.RelevanceEvaluator(qrels, _MEASURES).evaluate(run)["q0"]
    assert evaluated["map"] == pytest.approx(average_precision(ranking, grades))
    assert evaluated["recip_rank"] == pytest.approx(reciprocal_rank(ranking, grades))
    assert evaluated["recall_10"] == pytest.approx(recall_at_k(ranking, grades, 10))
    assert evaluated["P_10"] == pytest.approx(precision_at_k(ranking, grades, 10))
    assert ndcg_at_k(ranking, grades, 10) != pytest.approx(evaluated["ndcg_cut_10"])
