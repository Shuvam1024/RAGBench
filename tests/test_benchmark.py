import json
from pathlib import Path

import pytest

from ragstat.evaluation.models import Benchmark
from ragstat.evaluation.runner import load_benchmark


def question() -> dict[str, object]:
    return {
        "id": "q",
        "question": "What?",
        "expected_answer": "Answer",
        "relevant_document_ids": ["a.txt"],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"id": " "},
        {"question": "\n"},
        {"expected_answer": "   "},
        {"relevant_document_ids": []},
        {"relevant_document_ids": ["a", "a"]},
        {"relevant_document_ids": [""]},
        {"typo": "x"},
    ],
)
def test_invalid_questions(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Benchmark.model_validate({"questions": [{**question(), **changes}]})


def test_retrieval_only_questions_may_omit_answers() -> None:
    raw = question()
    raw["expected_answer"] = ""
    parsed = Benchmark.model_validate({"questions": [raw]})
    assert parsed.questions[0].grades() == {"a.txt": 1}
    raw["relevance_grades"] = {"a.txt": 2}
    graded = Benchmark.model_validate({"questions": [raw]})
    assert graded.questions[0].grades() == {"a.txt": 2}


def test_empty_duplicates_and_version() -> None:
    for raw in (
        {"questions": []},
        {"questions": [question(), question()]},
        {"schema_version": 2, "questions": [question()]},
    ):
        with pytest.raises(ValueError):
            Benchmark.model_validate(raw)


def test_unknown_document_and_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"questions": [question()]}))
    with pytest.raises(ValueError, match="unknown documents"):
        load_benchmark(path, {"b.txt"})
    assert load_benchmark(path, {"a.txt"}).questions[0].id == "q"
    path.write_text("{")
    with pytest.raises(ValueError, match="Cannot load benchmark"):
        load_benchmark(path, {"a.txt"})
