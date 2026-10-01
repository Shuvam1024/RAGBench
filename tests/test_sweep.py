import json
from pathlib import Path

from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.evaluation.sweep import load_sweep, run_sweep


def test_one_factor_sweep_changes_only_the_named_axis(tmp_path: Path) -> None:
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "a.txt").write_text("apple orchard apple orchard fruit basket", encoding="utf-8")
    (documents / "b.txt").write_text("ocean wave ocean wave tide pool", encoding="utf-8")
    benchmark = {
        "schema_version": 1,
        "questions": [
            {
                "id": "q1",
                "question": "apple",
                "expected_answer": "orchard",
                "relevant_document_ids": ["a.txt"],
            }
        ],
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    (tmp_path / "base.yaml").write_text(
        "dataset:\n  documents_path: docs\n  benchmark_path: benchmark.json\n"
        "chunking:\n  chunk_size: 4\n  overlap: 0\n"
        "retrieval:\n  type: bm25\n  k1: 1.5\n  b: 0.75\n"
        "evaluation:\n  recall_at_k: [1]\n",
        encoding="utf-8",
    )
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "base: base.yaml\n"
        "axes:\n"
        "  - name: chunk_size\n"
        "    field: chunking.chunk_size\n"
        "    values: [2, 4]\n"
        "  - name: k1\n"
        "    field: retrieval.k1\n"
        "    values: [1.5, 0.8]\n",
        encoding="utf-8",
    )
    _path, sweep, base = load_sweep(sweep_path)
    result = run_sweep(base, sweep, sweep.base)
    assert [row.name for row in result.rows] == ["base", "chunk_size=2", "k1=0.8"]
    assert result.rows[0].chunk_count != result.rows[1].chunk_count
    assert result.rows[0].metrics["mrr"] == result.rows[0].metrics["map"]
    assert {row.retriever for row in result.rows} == {"bm25"}
    output = tmp_path / "sweep.json"
    invoked = CliRunner().invoke(
        app, ["sweep", "--config", str(sweep_path), "--output", str(output)]
    )
    assert invoked.exit_code == 0, invoked.output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [row["name"] for row in saved["rows"]] == ["base", "chunk_size=2", "k1=0.8"]
    assert "MRR=" in invoked.output
