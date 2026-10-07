import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragstat.cli import app
from ragstat.config import load_config
from ragstat.evaluation.runner import evaluate
from ragstat.ingestion.chunker import Chunk
from ragstat.retrieval.base import Retriever, SearchResult


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("apple orchard apple orchard")
    (docs / "b.txt").write_text("ocean wave ocean wave")
    (docs / "c.txt").write_text("snow mountain snow mountain")
    benchmark = {
        "schema_version": 1,
        "questions": [
            {
                "id": "q1",
                "question": "apple",
                "expected_answer": "orchard",
                "relevant_document_ids": ["a.txt"],
            },
            {
                "id": "q2",
                "question": "ocean",
                "expected_answer": "wave",
                "relevant_document_ids": ["b.txt"],
            },
        ],
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark))
    config = tmp_path / "run.yaml"
    config.write_text(
        "dataset:\n  documents_path: docs\n  benchmark_path: benchmark.json\n"
        "chunking:\n  chunk_size: 2\n  overlap: 0\n"
        "evaluation:\n  recall_at_k: [1, 2]\n"
        "output:\n  json_path: default.json\n"
    )
    return config


def test_runner_repeatability_and_fingerprints(experiment: Path) -> None:
    config = load_config(experiment)
    first, second = evaluate(config), evaluate(config)
    assert first.corpus_sha256 == second.corpus_sha256
    assert first.mrr == second.mrr
    assert [q.retrieved_documents for q in first.questions] == [
        q.retrieved_documents for q in second.questions
    ]
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
    result = runner.invoke(
        app, ["evaluate", "--config", str(experiment), "--output", str(override)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(override.read_text())["config"]["output"]["json_path"] == str(override)


def test_cli_rejects_input_overwrite(experiment: Path, tmp_path: Path) -> None:
    for path in (
        experiment,
        tmp_path / "benchmark.json",
        tmp_path / "docs" / "a.txt",
        tmp_path / "docs" / "new.md",
    ):
        before = path.read_bytes() if path.exists() else None
        result = CliRunner().invoke(
            app, ["evaluate", "--config", str(experiment), "--output", str(path)]
        )
        assert result.exit_code == 1
        assert (path.read_bytes() if path.exists() else None) == before


def test_cli_errors(experiment: Path) -> None:
    result = CliRunner().invoke(
        app, ["evaluate", "--config", str(experiment.parent / "absent.yaml")]
    )
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


def test_support_fixture_hash_and_new_metrics_stay_consistent() -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    committed = json.loads((root / "benchmarks" / "baseline.json").read_text(encoding="utf-8"))
    result = evaluate(load_config(root / "configs" / "baseline.yaml"))
    assert result.corpus_sha256 == committed["corpus_sha256"]
    assert result.benchmark_sha256 == committed["benchmark_sha256"]
    assert result.mrr == pytest.approx(committed["mrr"])
    assert result.recall_at_k[1] == pytest.approx(committed["recall_at_k"]["1"])
    assert result.chunk_count == result.document_count
    assert result.multi_chunk_documents == 0
    assert result.mean_average_precision is not None
    assert result.ndcg_at_k is not None
    if importlib.util.find_spec("torch") is None:
        assert result.environment["pytorch_threads"] == "not-installed"
    else:
        import torch

        assert result.environment["pytorch_threads"] == str(torch.get_num_threads())


def test_stored_hits_do_not_change_full_ranking_metrics(experiment: Path) -> None:
    full = evaluate(load_config(experiment))
    text = experiment.read_text(encoding="utf-8").replace(
        "recall_at_k: [1, 2]\n", "recall_at_k: [1, 2]\n  stored_hits: 1\n"
    )
    experiment.write_text(text, encoding="utf-8")
    stored = evaluate(load_config(experiment))
    assert len(stored.questions[0].retrieved_documents) == 1
    assert stored.mrr == full.mrr
    assert stored.mean_average_precision == full.mean_average_precision
    assert stored.questions[0].ndcg_at_k == full.questions[0].ndcg_at_k


def test_generation_requires_reference_answers(experiment: Path) -> None:
    benchmark = experiment.parent / "benchmark.json"
    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    payload["questions"][0]["expected_answer"] = ""
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    experiment.write_text(
        experiment.read_text(encoding="utf-8") + "generation:\n  provider: extractive\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_answer"):
        evaluate(load_config(experiment))


def test_portable_paths_are_relative(
    experiment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "out.json"
    result = CliRunner().invoke(
        app,
        ["evaluate", "--config", str(experiment), "--output", str(destination), "--portable"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["config"]["dataset"]["documents_path"] == "docs"
    assert report["config"]["output"]["json_path"] == "out.json"


def test_document_ranks_collapse_duplicate_chunk_hits(experiment: Path) -> None:
    result = evaluate(load_config(experiment), DuplicateHits())
    assert result.questions[0].reciprocal_rank == 0.5
    assert result.questions[0].recall_at_k[2] == 1
    assert result.mrr == 0.75
