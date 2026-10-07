"""Fuse scores over the complete, aligned chunk corpus before selecting K."""

import math
from collections.abc import Sequence

from ragstat.ingestion.chunker import Chunk
from ragstat.retrieval.base import Retriever, SearchResult


def min_max_normalize(scores: Sequence[float]) -> list[float]:
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Normalization requires a nonempty finite score vector")
    low, high = min(scores), max(scores)
    if low == high:
        return [0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


class HybridRetriever(Retriever):
    def __init__(
        self,
        bm25: Retriever,
        dense: Retriever,
        dense_weight: float = 0.5,
        fusion: str = "minmax",
        rrf_k: int = 60,
    ) -> None:
        super().__init__()
        if (
            isinstance(dense_weight, bool)
            or not math.isfinite(dense_weight)
            or not 0 <= dense_weight <= 1
        ):
            raise ValueError("dense_weight must be between zero and one")
        if fusion not in {"minmax", "rrf"}:
            raise ValueError("fusion must be minmax or rrf")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        self.bm25, self.dense, self.dense_weight = bm25, dense, dense_weight
        self.fusion, self.rrf_k = fusion, rrf_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        validated = self.validate_chunks(chunks)
        # If either child fails, retrieval must not expose a partly replaced index.
        self._chunks = ()
        self.bm25.index(validated)
        self.dense.index(validated)
        self._chunks = validated

    def _aligned_scores(
        self, results: Sequence[SearchResult], allow_empty: bool = False
    ) -> list[float]:
        if not results and allow_empty:
            return [0.0] * len(self._chunks)
        scores = {result.chunk.id: result.score for result in results}
        expected = {chunk.id for chunk in self._chunks}
        if len(scores) != len(results) or set(scores) != expected:
            raise ValueError("Hybrid retrievers must return exactly the same indexed chunk IDs")
        return [scores[chunk.id] for chunk in self._chunks]

    def _rrf_scores(self, results: Sequence[SearchResult], *, allow_empty: bool) -> list[float]:
        if not results and allow_empty:
            return [0.0] * len(self._chunks)
        positions = {chunk.id: index for index, chunk in enumerate(self._chunks)}
        seen = {result.chunk.id for result in results}
        if len(seen) != len(results) or seen != set(positions):
            raise ValueError("Hybrid retrievers must return exactly the same indexed chunk IDs")
        scores = [0.0] * len(self._chunks)
        for rank, result in enumerate(results, start=1):
            scores[positions[result.chunk.id]] += 1.0 / (self.rrf_k + rank)
        return scores

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.validate_query(k)
        if not query.strip():
            return []
        size = len(self._chunks)
        if self.fusion == "rrf":
            # Tied chunk scores are already broken by chunk ID, so they receive distinct ranks.
            lexical = self._rrf_scores(self.bm25.retrieve(query, size), allow_empty=True)
            dense = self._rrf_scores(self.dense.retrieve(query, size), allow_empty=False)
            scores = [left + right for left, right in zip(lexical, dense, strict=True)]
            return self.rank(scores, k)
        lexical = min_max_normalize(
            self._aligned_scores(self.bm25.retrieve(query, size), allow_empty=True)
        )
        dense = min_max_normalize(self._aligned_scores(self.dense.retrieve(query, size)))
        scores = [
            (1 - self.dense_weight) * left + self.dense_weight * right
            for left, right in zip(lexical, dense, strict=True)
        ]
        return self.rank(scores, k)

    @property
    def metadata(self) -> dict[str, str | int]:
        details: dict[str, str | int] = {
            **self.dense.metadata,
            "implementation": type(self).__name__,
            "fusion": self.fusion,
        }
        if self.fusion == "rrf":
            details["rrf_k"] = self.rrf_k
        tokenizer = self.bm25.metadata.get("tokenizer")
        if isinstance(tokenizer, str):
            details["bm25_tokenizer"] = tokenizer
        return details
