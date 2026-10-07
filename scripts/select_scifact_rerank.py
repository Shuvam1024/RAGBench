"""Choose a cross-encoder candidate depth on the SciFact train split.

The test split is not read. The first stage is the frozen hybrid in
``configs/scifact-rerank-train.yaml``. The rule is fixed here:

- Score candidate depths 20, 50, and 100. The cross-encoder runs once on the
  widest set. Smaller depths rerank a prefix of that first-stage list.
- The winner maximizes train nDCG@10. Ties break toward the smaller depth.
"""

import json
from pathlib import Path

from ragstat.config import HybridConfig, load_config
from ragstat.evaluation.runner import evaluate, evaluate_candidate_depths, relativize_result
from ragstat.evaluation.train_selection import choose_candidate_k

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BENCHMARK = ROOT / ".cache/beir/scifact/benchmark.train.json"
CONFIG = ROOT / "configs/scifact-rerank-train.yaml"
DEPTHS = (20, 50, 100)


def _metrics(result: object) -> dict[str, float]:
    assert hasattr(result, "ndcg_at_k") and hasattr(result, "recall_at_k")
    ndcg = result.ndcg_at_k or {}
    precision = result.precision_at_k or {}
    return {
        "ndcg@10": ndcg[10],
        "recall@10": result.recall_at_k[10],
        "recall@100": result.recall_at_k[100],
        "precision@10": precision[10],
        "map": result.mean_average_precision,
        "mrr": result.mrr,
        "candidate_recall_at_100": result.candidate_recall_at_100,
    }


def main() -> None:
    config = load_config(CONFIG)
    benchmark = config.dataset.benchmark_path
    if not TRAIN_BENCHMARK.is_file() or benchmark.resolve() != TRAIN_BENCHMARK.resolve():
        raise SystemExit(f"{CONFIG} must point at the SciFact train benchmark")
    retrieval = config.retrieval
    if (
        config.rerank is None
        or config.rerank.candidate_k != max(DEPTHS)
        or not isinstance(retrieval, HybridConfig)
        or retrieval.fusion != "minmax"
        or retrieval.dense_weight != 0.5
        or retrieval.bm25.tokenizer != "stem"
        or config.chunking.chunk_size != 480
        or config.chunking.unit != "words"
    ):
        raise SystemExit("refusing to select: the train file is not the frozen hybrid at depth 100")
    print("Scoring train candidate depths 20, 50, and 100", flush=True)
    reports = evaluate_candidate_depths(
        config,
        DEPTHS,
        progress=lambda done, total: (
            print(f"train questions {done}/{total}", flush=True)
            if done % 50 == 0 or done == total
            else None
        ),
    )
    rows = [
        {
            "candidate_k": depth,
            "question_count": reports[depth].question_count,
            "metrics": _metrics(reports[depth]),
        }
        for depth in DEPTHS
    ]
    winner = choose_candidate_k(
        [(int(row["candidate_k"]), float(row["metrics"]["ndcg@10"])) for row in rows]
    )
    if winner == max(DEPTHS):
        official = reports[winner]
    else:
        # The grid clock scored the widest pool. Time the winning depth on its own.
        assert config.rerank is not None
        official = evaluate(
            config.model_copy(
                update={"rerank": config.rerank.model_copy(update={"candidate_k": winner})}
            )
        )
    payload = {
        "split": "train",
        "question_count": reports[DEPTHS[0]].question_count,
        "selection_metric": "ndcg@10",
        "tie_break": ["smaller candidate_k"],
        "first_stage": "frozen hybrid dense_weight 0.5, stem, chunk 480",
        "model_name": config.rerank.model_name,
        "model_revision": config.rerank.revision,
        "max_length": config.rerank.max_length,
        "text": config.rerank.text,
        "candidates": rows,
        "winner": {"candidate_k": winner, "metrics": _metrics(official)},
    }
    destination = ROOT / "results" / "rerank-train-selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / "results" / "rerank-train.json"
    report_path.write_text(
        relativize_result(official, ROOT).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"winner candidate_k={winner}")
    print(f"Saved: {destination}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
