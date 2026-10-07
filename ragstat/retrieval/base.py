"""A shared lifecycle and deterministic ordering for chunk retrieval."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ragstat.ingestion.chunker import Chunk


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk: Chunk
    score: float = Field(allow_inf_nan=False)


class Retriever(ABC):
    def __init__(self) -> None:
        self._chunks: tuple[Chunk, ...] = ()

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Replace the index with the supplied chunks."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        """Return at most K chunks, with higher scores first."""

    @property
    def metadata(self) -> dict[str, str | int]:
        return {"implementation": type(self).__name__}

    @staticmethod
    def validate_chunks(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
        result = tuple(chunks)
        if not result:
            raise ValueError("Cannot index an empty chunk collection")
        if len({chunk.id for chunk in result}) != len(result):
            raise ValueError("Chunk IDs must be unique")
        return result

    def validate_query(self, k: int) -> None:
        if not self._chunks:
            raise ValueError("Index the retriever before searching")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

    def rank(self, scores: Sequence[float], k: int) -> list[SearchResult]:
        if len(scores) != len(self._chunks) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Expected one finite score per indexed chunk")
        pairs = sorted(
            zip(self._chunks, scores, strict=True), key=lambda pair: (-pair[1], pair[0].id)
        )
        return [SearchResult(chunk=chunk, score=float(score)) for chunk, score in pairs[:k]]
