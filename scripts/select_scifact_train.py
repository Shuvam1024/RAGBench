"""Choose chunked BM25 settings on the SciFact train split.

The test split is not read. The rule is fixed in this file:

- Candidates use word windows, overlap 20, k1 1.5, and b 0.75.
  Tokenizer is ``whitespace`` or ``stem``. Chunk size is 60, 120, 240, or 480.
- The winner maximizes train nDCG@10. Ties break toward the smaller chunk
  size, then ``whitespace`` before ``stem``.
- Document-level BM25 from ``configs/scifact-bm25-document-train.yaml`` is
  scored and stored as a reference. It is not eligible to win.
"""

import json
from pathlib import Path

from ragbench.config import load_config
from ragbench.evaluation.runner import evaluate
from ragbench.evaluation.train_selection import selection_key

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BENCHMARK = ROOT / ".cache/beir/scifact/benchmark.train.json"
CANDIDATE_BASE = ROOT / "configs/scifact-bm25-train.yaml"
DOCUMENT_REFERENCE = ROOT / "configs/scifact-bm25-document-train.yaml"
CHUNK_SIZES = (60, 120, 240, 480)
TOKENIZERS = ("whitespace", "stem")


def _metrics(result: object) -> dict[str, float]:
    assert hasattr(result, "ndcg_at_k")
    ndcg = result.ndcg_at_k or {}
    return {
        "ndcg@10": ndcg[10],
        "recall@10": result.recall_at_k[10],
        "recall@100": result.recall_at_k[100],
        "precision@10": (result.precision_at_k or {})[10],
        "map": result.mean_average_precision,
        "mrr": result.mrr,
    }


def _refuse_test_split(config_path: Path) -> None:
    config = load_config(config_path)
    benchmark = config.dataset.benchmark_path
    if benchmark.name == "benchmark.json" or not TRAIN_BENCHMARK.is_file():
        raise SystemExit(f"{config_path} must point at the SciFact train benchmark")
    if benchmark.resolve() != TRAIN_BENCHMARK.resolve():
        raise SystemExit(f"{config_path} benchmark is {benchmark}, not the train split")


def main() -> None:
    _refuse_test_split(CANDIDATE_BASE)
    _refuse_test_split(DOCUMENT_REFERENCE)
    base = load_config(CANDIDATE_BASE)
    rows: list[dict[str, object]] = []
    for tokenizer in TOKENIZERS:
        for chunk_size in CHUNK_SIZES:
            config = base.model_copy(
                update={
                    "chunking": base.chunking.model_copy(update={"chunk_size": chunk_size}),
                    "retrieval": base.retrieval.model_copy(update={"tokenizer": tokenizer}),
                }
            )
            result = evaluate(config)
            rows.append(
                {
                    "tokenizer": tokenizer,
                    "chunk_size": chunk_size,
                    "overlap": config.chunking.overlap,
                    "k1": config.retrieval.k1,
                    "b": config.retrieval.b,
                    "unit": config.chunking.unit,
                    "question_count": result.question_count,
                    "chunk_count": result.chunk_count,
                    "multi_chunk_documents": result.multi_chunk_documents,
                    "metrics": _metrics(result),
                }
            )
            print(
                f"{tokenizer} chunk={chunk_size} nDCG@10={rows[-1]['metrics']['ndcg@10']:.4f}",
                flush=True,
            )
    winner = min(rows, key=selection_key)
    reference_result = evaluate(load_config(DOCUMENT_REFERENCE))
    payload = {
        "split": "train",
        "question_count": rows[0]["question_count"],
        "selection_metric": "ndcg@10",
        "tie_break": ["smaller chunk_size", "whitespace before stem"],
        "candidates": rows,
        "winner": {
            "tokenizer": winner["tokenizer"],
            "chunk_size": winner["chunk_size"],
            "overlap": winner["overlap"],
            "k1": winner["k1"],
            "b": winner["b"],
            "unit": winner["unit"],
            "metrics": winner["metrics"],
        },
        "document_level_reference": {
            "eligible": False,
            "config": "configs/scifact-bm25-document-train.yaml",
            "unit": "document",
            "tokenizer": "stem",
            "k1": 0.9,
            "b": 0.4,
            "question_count": reference_result.question_count,
            "chunk_count": reference_result.chunk_count,
            "metrics": _metrics(reference_result),
        },
    }
    destination = ROOT / "results" / "scifact-bm25-train-selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"winner tokenizer={winner['tokenizer']} chunk_size={winner['chunk_size']}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
