import json
from pathlib import Path

import pytest

from ragbench.evaluation.models import Benchmark
from ragbench.evaluation.runner import load_benchmark


def question() -> dict[str, object]:
    return {"id": "q", "question": "What?", "expected_answer": "Answer", "relevant_document_ids": ["a.txt"]}


@pytest.mark.parametrize("changes", [
    {"id": " "}, {"question": "\n"}, {"expected_answer": ""}, {"relevant_document_ids": []},
    {"relevant_document_ids": ["a", "a"]}, {"relevant_document_ids": [""]}, {"typo": "x"},
])
def test_invalid_questions(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Benchmark.model_validate({"questions": [{**question(), **changes}]})


def test_empty_duplicates_and_version() -> None:
    for raw in ({"questions": []}, {"questions": [question(), question()]},
                {"schema_version": 2, "questions": [question()]}):
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
