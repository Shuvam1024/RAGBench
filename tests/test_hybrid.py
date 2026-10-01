from collections.abc import Callable, Sequence

import pytest

from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.base import Retriever, SearchResult
from ragbench.retrieval.hybrid import HybridRetriever, min_max_normalize


class ScoredRetriever(Retriever):
    def __init__(self, scores: Sequence[float], empty: bool = False) -> None:
        super().__init__()
        self.scores, self.empty = scores, empty

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = self.validate_chunks(chunks)

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        return [] if self.empty else self.rank(self.scores, k)


def test_normalization() -> None:
    assert min_max_normalize([-2, 0, 2]) == [0, 0.5, 1]
    assert min_max_normalize([-3, -3]) == [0, 0]
    assert min_max_normalize([100, 100]) == [0, 0]
    for scores in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            min_max_normalize(scores)


@pytest.mark.parametrize(
    ("weight", "order", "scores"),
    [
        (0.0, ["a", "b", "c"], [1, 0.5, 0]),
        (1.0, ["c", "b", "a"], [1, 0.8, 0]),
        (0.5, ["b", "a", "c"], [0.65, 0.5, 0.5]),
    ],
)
def test_manual_fusion(
    make_chunk: Callable[..., Chunk], weight: float, order: list[str], scores: list[float]
) -> None:
    retriever = HybridRetriever(ScoredRetriever([10, 5, 0]), ScoredRetriever([-1, 0.6, 1]), weight)
    retriever.index([make_chunk("text", name) for name in ("a", "b", "c")])
    hits = retriever.retrieve("query", 10)
    assert [hit.chunk.id for hit in hits] == order
    assert [hit.score for hit in hits] == pytest.approx(scores)
    assert retriever.retrieve("query", 1) == hits[:1]
    assert retriever.retrieve(" ", 1) == []


def test_empty_lexical_and_constant_vectors(make_chunk: Callable[..., Chunk]) -> None:
    chunks = [make_chunk("text", name) for name in ("b", "a")]
    retriever = HybridRetriever(ScoredRetriever([], empty=True), ScoredRetriever([4, 4]))
    retriever.index(chunks)
    hits = retriever.retrieve("!!!", 2)
    assert [hit.chunk.id for hit in hits] == ["a", "b"]
    assert [hit.score for hit in hits] == [0, 0]


def test_misalignment_and_bad_weights(make_chunk: Callable[..., Chunk]) -> None:
    for weight in (-1, 2, float("nan"), True):
        with pytest.raises(ValueError):
            HybridRetriever(ScoredRetriever([1]), ScoredRetriever([1]), weight)
    left, right = ScoredRetriever([1]), ScoredRetriever([1])
    retriever = HybridRetriever(left, right)
    retriever.index([make_chunk("text", "a")])
    right.index([make_chunk("text", "wrong")])
    with pytest.raises(ValueError, match="same indexed chunk IDs"):
        retriever.retrieve("query", 1)
