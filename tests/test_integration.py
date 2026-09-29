"""Real model CLI checks run in fresh processes to isolate native runtimes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_real_model_semantic_search(mode: str, tmp_path: Path) -> None:
    for dependency in ("faiss", "sentence_transformers"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Install the dense extra for {dependency}")
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in {
        "orchard": "An orchard grows apples and pears.",
        "ocean": "Whales and dolphins swim in the ocean.",
        "mountain": "Snow covers the tops of tall mountains.",
    }.items():
        (docs / f"{name}.txt").write_text(text)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "sea",
                        "question": "Which animals live in the sea?",
                        "expected_answer": "Whales and dolphins.",
                        "relevant_document_ids": ["ocean.txt"],
                    }
                ]
            }
        )
    )
    source = Path(__file__).resolve().parents[1] / "configs" / f"{mode}.yaml"
    config = yaml.safe_load(source.read_text())
    config["dataset"] = {"documents_path": str(docs), "benchmark_path": str(benchmark)}
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))
    report_path = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragbench",
            "evaluate",
            "--config",
            str(config_path),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(report_path.read_text())
    assert report["questions"][0]["retrieved_documents"][0]["document_id"] == "ocean.txt"
    assert report["mrr"] == 1
    assert len(report["retriever"]["model_revision"]) == 40
