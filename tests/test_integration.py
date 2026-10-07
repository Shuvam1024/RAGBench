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
            "ragstat",
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


@pytest.mark.integration
def test_real_cross_encoder_reports_candidate_recall_and_pipeline(tmp_path: Path) -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("Install the dense extra for sentence_transformers")
    source = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "scifact-rerank-train.yaml").read_text()
    )
    rerank = source["rerank"]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "apples.txt").write_text("An orchard grows apples and pears.")
    (docs / "ocean.txt").write_text("Whales and dolphins swim in the ocean.")
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
    config = {
        "dataset": {"documents_path": str(docs), "benchmark_path": str(benchmark)},
        "retrieval": {"type": "bm25"},
        "rerank": rerank,
        "evaluation": {"recall_at_k": [1, 100], "ndcg_at_k": [10], "stored_hits": 2},
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))
    report_path = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragstat",
            "evaluate",
            "--config",
            str(config_path),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(report_path.read_text())
    assert report["retriever"]["reranker_revision"] == rerank["revision"]
    assert report["retriever"]["reranker_text"] == "best_first_stage_chunk"
    assert report["retriever"]["reranker_max_length"] == 512
    assert report["candidate_recall_at_100"] == 1
    assert report["timings"]["rerank_p95_ms"] >= 0
    assert report["timings"]["pipeline_p95_ms"] >= report["timings"]["retrieval_p95_ms"]
    assert "Candidate Recall@100" in process.stdout
    assert "Top-K pipeline p95" in process.stdout


@pytest.mark.integration
def test_real_nli_maps_entailment_and_contradiction() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("Install the dense extra for sentence_transformers")
    from ragstat.evaluation.nli import CrossEncoderNli, NliModelConfig, classify_nli

    config = NliModelConfig()
    scorer = CrossEncoderNli(config)
    scores = scorer.score(
        [
            ("A dog is a mammal.", "A dog is an animal."),
            ("A dog is a mammal.", "A dog is a reptile."),
        ]
    )
    assert classify_nli(scores[0])[0] == "SUPPORT"
    assert classify_nli(scores[1])[0] == "CONTRADICT"
    assert scorer.metadata["nli_revision"] == config.revision
    total = scores[0].contradiction + scores[0].entailment + scores[0].neutral
    assert total == pytest.approx(1.0, abs=1e-5)
