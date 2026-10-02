"""Pure document-level retrieval metrics, with duplicate hits removed first."""

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean


def unique_documents(ranked_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ranked_ids))


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant document IDs must not be empty")
    _cutoff(k)
    return len(set(unique_documents(ranked_ids)[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant document IDs must not be empty")
    for rank, document_id in enumerate(unique_documents(ranked_ids), start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]], relevance: Sequence[Iterable[str]]
) -> float:
    if not rankings or len(rankings) != len(relevance):
        raise ValueError("Provide equally sized, nonempty rankings and relevance collections")
    return fmean(
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(rankings, relevance, strict=True)
    )


def _cutoff(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top K document slots that are relevant. The denominator is K."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant document IDs must not be empty")
    _cutoff(k)
    return len(set(unique_documents(ranked_ids)[:k]) & relevant) / k


def average_precision(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Binary average precision over the full unique-document ranking."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant document IDs must not be empty")
    hits = 0
    total = 0.0
    for rank, document_id in enumerate(unique_documents(ranked_ids), start=1):
        if document_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def _dcg(grades: Sequence[float]) -> float:
    total = 0.0
    for rank, grade in enumerate(grades, start=1):
        total += (2.0**grade - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    """Normalized DCG at K. Unlisted documents have grade 0. Grades must be positive."""
    if not grades or any(isinstance(grade, bool) or grade <= 0 for grade in grades.values()):
        raise ValueError("grades must be a nonempty map of positive relevance values")
    _cutoff(k)
    gains = [
        float(grades.get(document_id, 0.0)) for document_id in unique_documents(ranked_ids)[:k]
    ]
    ideal = _dcg(sorted((float(grade) for grade in grades.values()), reverse=True)[:k])
    if ideal == 0:
        raise ValueError("Ideal gain must be positive")
    return _dcg(gains) / ideal
