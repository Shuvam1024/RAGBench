"""Lock hybrid dense_weight from a SciFact train-split sweep.

The test split is not read. The base row's weight comes from the sweep's base
config. The winner maximizes train nDCG@10, then prefers the smaller weight.
"""

import json
from pathlib import Path

from ragstat.config import HybridConfig, load_config
from ragstat.evaluation.models import Benchmark
from ragstat.evaluation.runner import fingerprint
from ragstat.evaluation.train_selection import choose_dense_weight

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BENCHMARK = ROOT / ".cache/beir/scifact/benchmark.train.json"
SWEEP_JSON = ROOT / "results" / "hybrid-train-sweep-c480.json"
BASE_CONFIG = ROOT / "configs" / "scifact-hybrid-train.yaml"


def measured_rows(payload: dict[str, object], base_weight: float) -> list[tuple[float, float]]:
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise SystemExit("Sweep JSON rows must be a list")
    rows: list[tuple[float, float]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise SystemExit("Sweep row must be an object")
        value = row["value"]
        metrics = row["metrics"]
        if not isinstance(metrics, dict):
            raise SystemExit("Sweep row metrics must be an object")
        weight = base_weight if value == "base" else float(value)
        rows.append((weight, float(metrics["ndcg@10"])))
    return rows


def main() -> None:
    if not SWEEP_JSON.is_file():
        raise SystemExit(f"Missing train sweep: {SWEEP_JSON}")
    if not TRAIN_BENCHMARK.is_file():
        raise SystemExit(f"Missing train benchmark: {TRAIN_BENCHMARK}")
    base = load_config(BASE_CONFIG)
    if base.dataset.benchmark_path.resolve() != TRAIN_BENCHMARK.resolve():
        raise SystemExit(f"{BASE_CONFIG} is not the SciFact train split")
    if not isinstance(base.retrieval, HybridConfig):
        raise SystemExit(f"{BASE_CONFIG} is not hybrid")
    payload = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    benchmark = Benchmark.model_validate_json(TRAIN_BENCHMARK.read_text(encoding="utf-8"))
    train_fingerprint = fingerprint(benchmark.model_dump(mode="json", exclude_none=True))
    if payload.get("benchmark_sha256") != train_fingerprint or len(benchmark.questions) != 809:
        raise SystemExit("Sweep fingerprint is not the SciFact train benchmark")
    if payload.get("base_config") != BASE_CONFIG.name:
        raise SystemExit("Sweep base_config does not match the train hybrid config")
    rows = measured_rows(payload, base.retrieval.dense_weight)
    weight = choose_dense_weight(rows)
    destination = ROOT / "results" / "scifact-hybrid-weight-train.json"
    body = {
        "split": "train",
        "selection_metric": "ndcg@10",
        "tie_break": "smaller dense_weight",
        "benchmark_sha256": payload["benchmark_sha256"],
        "lexical": {
            "tokenizer": base.retrieval.bm25.tokenizer,
            "chunk_size": base.chunking.chunk_size,
            "overlap": base.chunking.overlap,
            "k1": base.retrieval.bm25.k1,
            "b": base.retrieval.bm25.b,
            "unit": base.chunking.unit,
        },
        "dense_weight": weight,
        "rows": [{"dense_weight": item[0], "ndcg@10": item[1]} for item in rows],
    }
    destination.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"dense_weight={weight}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
