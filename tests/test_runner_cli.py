import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config import load_config
from ragbench.evaluation.runner import evaluate
from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.base import Retriever, SearchResult


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("apple orchard apple orchard")
    (docs / "b.txt").write_text("ocean wave ocean wave")
    (docs / "c.txt").write_text("snow mountain snow mountain")
    benchmark = {"schema_version": 1, "questions": [
        {"id": "q1", "question": "apple", "expected_answer": "orchard", "relevant_document_ids": ["a.txt"]},
        {"id": "q2", "question": "ocean", "expected_answer": "wave", "relevant_document_ids": ["b.txt"]},
    ]}
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark))
    config = tmp_path / "run.yaml"
    config.write_text("dataset:\n  documents_path: docs\n  benchmark_path: benchmark.json\n"
                      "chunking:\n  chunk_size: 2\n  overlap: 0\n"
                      "evaluation:\n  recall_at_k: [1, 2]\n"
                      "output:\n  json_path: default.json\n")
    return config


def test_runner_repeatability_and_fingerprints(experiment: Path) -> None:
    config = load_config(experiment)
    first, second = evaluate(config), evaluate(config)
    assert first == second
    assert first.mrr == 1
    assert first.recall_at_k == {1: 1, 2: 1}
    assert first.chunk_count == 6
    assert len(first.questions[0].retrieved_documents) == 3
    (experiment.parent / "docs" / "a.txt").write_text("apple orchard apple fruit")
    assert evaluate(config).corpus_sha256 != first.corpus_sha256


def test_cli_output_override_and_json(experiment: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "--config", str(experiment)])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "default.json").read_text())
    assert report["mrr"] == 1
    assert "MRR        1.0000" in result.output
    assert report["questions"][0]["retrieved_documents"][0]["document_id"] == "a.txt"
    override = tmp_path / "nested" / "override.json"
    result = runner.invoke(app, ["evaluate", "--config", str(experiment), "--output", str(override)])
    assert result.exit_code == 0, result.output
    assert json.loads(override.read_text())["config"]["output"]["json_path"] == str(override)


def test_cli_rejects_input_overwrite(experiment: Path, tmp_path: Path) -> None:
    for path in (experiment, tmp_path / "benchmark.json", tmp_path / "docs" / "a.txt",
                 tmp_path / "docs" / "new.md"):
        before = path.read_bytes() if path.exists() else None
        result = CliRunner().invoke(app, ["evaluate", "--config", str(experiment), "--output", str(path)])
        assert result.exit_code == 1
        assert (path.read_bytes() if path.exists() else None) == before


def test_cli_errors(experiment: Path) -> None:
    result = CliRunner().invoke(app, ["evaluate", "--config", str(experiment.parent / "absent.yaml")])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output
    (experiment.parent / "benchmark.json").write_text("{}")
    result = CliRunner().invoke(app, ["evaluate", "--config", str(experiment)])
    assert result.exit_code == 1
    assert not (experiment.parent / "default.json").exists()


class DuplicateHits(Retriever):
    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = self.validate_chunks(chunks)

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        # B occupies two chunk positions, but only one document position.
        order = {"b.txt": 3, "a.txt": 2, "c.txt": 1}
        return self.rank([float(order[chunk.document_id]) for chunk in self._chunks], k)


def test_document_ranks_collapse_duplicate_chunk_hits(experiment: Path) -> None:
    result = evaluate(load_config(experiment), DuplicateHits())
    assert result.questions[0].reciprocal_rank == 0.5
    assert result.questions[0].recall_at_k[2] == 1
    assert result.mrr == 0.75
