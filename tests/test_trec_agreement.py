"""Cross-check retrieval metrics against a committed pytrec_eval fixture.

Each query stores pytrec_eval's per-query scores and ragstat's scores.
Binary queries must match pytrec_eval. The graded query stores both values
because pytrec_eval's ``ndcg_cut`` uses the grade as the gain and ragstat
uses ``2**grade - 1``.
"""

import json
from pathlib import Path

import pytest

from ragstat.evaluation.retrieval import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytrec_eval = pytest.importorskip("pytrec_eval")

FIXTURE = Path(__file__).parent / "fixtures" / "pytrec_agreement.json"
_MEASURES = {"map", "recip_rank", "ndcg_cut_10", "recall_10", "P_10"}
_OURS = {
    "map": average_precision,
    "recip_rank": reciprocal_rank,
    "ndcg_at_10": lambda ranking, grades: ndcg_at_k(ranking, grades, 10),
    "recall_10": lambda ranking, grades: recall_at_k(ranking, grades, 10),
    "P_10": lambda ranking, grades: precision_at_k(ranking, grades, 10),
}


def _queries() -> list[dict[str, object]]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not loaded:
        raise AssertionError("pytrec fixture must be a nonempty list")
    return loaded


def test_committed_fixture_matches_pytrec_eval_per_query() -> None:
    queries = _queries()
    qrels = {str(query["id"]): query["grades"] for query in queries}
    run = {
        str(query["id"]): {
            doc_id: float(len(query["ranking"]) - rank)
            for rank, doc_id in enumerate(query["ranking"])
        }
        for query in queries
    }
    evaluated = pytrec_eval.RelevanceEvaluator(qrels, _MEASURES).evaluate(run)
    for query in queries:
        query_id = str(query["id"])
        expected = query["pytrec"]
        assert isinstance(expected, dict)
        for measure in _MEASURES:
            assert evaluated[query_id][measure] == pytest.approx(expected[measure])


def test_committed_fixture_matches_ragstat_per_query() -> None:
    for query in _queries():
        ranking = query["ranking"]
        grades = query["grades"]
        expected = query["ragstat"]
        assert isinstance(ranking, list)
        assert isinstance(grades, dict)
        assert isinstance(expected, dict)
        for name, function in _OURS.items():
            assert function(ranking, grades) == pytest.approx(expected[name])
        binary = all(grade == 1 for grade in grades.values())
        pytrec_ndcg = query["pytrec"]["ndcg_cut_10"]
        if binary:
            assert expected["ndcg_at_10"] == pytest.approx(pytrec_ndcg)
        else:
            assert expected["ndcg_at_10"] != pytest.approx(pytrec_ndcg)
