"""Pure document-level retrieval metrics, with duplicate hits removed first."""

from collections.abc import Iterable, Sequence
from statistics import fmean


def unique_documents(ranked_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ranked_ids))


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant document IDs must not be empty")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return len(set(unique_documents(ranked_ids)[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant document IDs must not be empty")
    for rank, document_id in enumerate(unique_documents(ranked_ids), start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: Sequence[Sequence[str]], relevance: Sequence[Iterable[str]]) -> float:
    if not rankings or len(rankings) != len(relevance):
        raise ValueError("Provide equally sized, nonempty rankings and relevance collections")
    return fmean(reciprocal_rank(ranked, relevant) for ranked, relevant in zip(rankings, relevance, strict=True))
