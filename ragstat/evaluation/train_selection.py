"""Pre-specified ordering for SciFact train-split selection.

Lower keys sort first. nDCG@10 is maximized. BM25 ties prefer the smaller
chunk size, then the whitespace tokenizer. Hybrid ties prefer the smaller
``dense_weight``.
"""

from collections.abc import Sequence


def selection_key(row: dict[str, object]) -> tuple[float, int, int]:
    metrics = row["metrics"]
    chunk_size = row["chunk_size"]
    if not isinstance(metrics, dict) or not isinstance(chunk_size, int):
        raise TypeError("a candidate row needs an int chunk_size and a metrics mapping")
    score = metrics.get("ndcg@10")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError("ndcg@10 must be a real number")
    return (-float(score), chunk_size, 0 if row["tokenizer"] == "whitespace" else 1)


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a real number")
    return float(value)


def choose_dense_weight(rows: Sequence[tuple[float, float]]) -> float:
    """Pick a train-split hybrid weight.

    Each row is ``(dense_weight, ndcg@10)``. The highest nDCG@10 wins. Equal
    scores prefer the smaller weight.
    """
    if not rows:
        raise ValueError("dense-weight selection needs at least one row")
    scored = [(_real(weight, "dense_weight"), _real(ndcg, "ndcg@10")) for weight, ndcg in rows]
    return min(scored, key=lambda item: (-item[1], item[0]))[0]


def choose_candidate_k(rows: Sequence[tuple[int, float]]) -> int:
    """Pick a train-split cross-encoder candidate depth.

    Each row is ``(candidate_k, ndcg@10)``. The highest nDCG@10 wins. Equal
    scores prefer the smaller depth.
    """
    if not rows:
        raise ValueError("candidate-depth selection needs at least one row")
    scored: list[tuple[int, float]] = []
    for depth, ndcg in rows:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise TypeError("candidate_k must be a positive integer")
        scored.append((depth, _real(ndcg, "ndcg@10")))
    return min(scored, key=lambda item: (-item[1], item[0]))[0]
