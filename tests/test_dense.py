from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("faiss")

from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.dense import DenseRetriever, normalize_vectors


class FixedEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    @property
    def metadata(self) -> dict[str, str | int]:
        return {"model_name": "fixed-test-vectors"}


def test_real_faiss_cosine_ranking_and_index_once(make_chunk: Callable[..., Chunk]) -> None:
    embedder = FixedEmbeddings({"east": [10, 0], "north": [0, 2], "west": [-3, 0], "query": [2, 1]})
    retriever = DenseRetriever(embedder=embedder)
    retriever.index([make_chunk("east", "e"), make_chunk("north", "n"), make_chunk("west", "w")])
    hits = retriever.retrieve("query", 10)
    assert [hit.chunk.id for hit in hits] == ["e", "n", "w"]
    assert [hit.score for hit in hits] == pytest.approx([2 / np.sqrt(5), 1 / np.sqrt(5), -2 / np.sqrt(5)])
    assert retriever.retrieve("query", 1) == hits[:1]
    assert embedder.calls == [["east", "north", "west"], ["query"], ["query"]]
    assert retriever.retrieve("  ", 1) == []


def test_dense_lifecycle_ties_and_dimension_mismatch(make_chunk: Callable[..., Chunk]) -> None:
    embedder = FixedEmbeddings({"same": [1, 1], "query": [1, 0], "bad": [1, 0, 0]})
    retriever = DenseRetriever(embedder=embedder)
    with pytest.raises(ValueError, match="before searching"):
        retriever.retrieve("query", 1)
    with pytest.raises(ValueError, match="empty"):
        retriever.index([])
    retriever.index([make_chunk("same", "z"), make_chunk("same", "a")])
    assert retriever.retrieve("query", 1)[0].chunk.id == "a"
    with pytest.raises(ValueError, match="dimension"):
        retriever.retrieve("bad", 1)
    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("query", 0)
    retriever.index([make_chunk("same", "new")])
    assert [hit.chunk.id for hit in retriever.retrieve("query", 10)] == ["new"]
    with pytest.raises(ValueError, match="unique"):
        retriever.index([make_chunk("same"), make_chunk("same")])


@pytest.mark.parametrize("values", [[[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]],
                                    [1, 2], [[]], [[1, 0], [0, 1]]])
def test_invalid_vectors(values: list[object]) -> None:
    with pytest.raises(ValueError):
        normalize_vectors(np.asarray(values, dtype=np.float32), rows=1)


def test_large_and_small_vectors_are_normalized_without_overflow() -> None:
    values = np.asarray([[1e30, 1e30], [1e-30, 1e-30]], dtype=np.float32)
    assert np.linalg.norm(normalize_vectors(values, 2), axis=1) == pytest.approx([1, 1])
