from collections.abc import Callable

import pytest

from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.bm25 import BM25Retriever


def test_keyword_ranking_and_case(make_chunk: Callable[..., Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index([make_chunk("apple orchard", "a"), make_chunk("ocean wave", "b"),
                     make_chunk("mountain snow", "c")])
    hits = retriever.retrieve("OCEAN!", 10)
    assert hits[0].chunk.id == "b"
    assert hits[0].score > hits[1].score
    assert hits == retriever.retrieve("ocean", 10)
    assert len(hits) == 3
    assert retriever.retrieve("!!!", 1) == []
    assert retriever.retrieve(" ", 1) == []
    assert [hit.chunk.id for hit in retriever.retrieve("absent", 3)] == ["a", "b", "c"]


def test_lifecycle_ties_and_negative_scores(make_chunk: Callable[..., Chunk]) -> None:
    retriever = BM25Retriever()
    with pytest.raises(ValueError, match="before searching"):
        retriever.retrieve("word", 1)
    with pytest.raises(ValueError, match="empty"):
        retriever.index([])
    a = make_chunk("word", "a")
    with pytest.raises(ValueError, match="unique"):
        retriever.index([a, a])
    with pytest.raises(ValueError, match="searchable"):
        retriever.index([make_chunk("!!!")])
    retriever.index([make_chunk("word", "b"), a])
    hits = retriever.retrieve("word", 2)
    assert [hit.chunk.id for hit in hits] == ["a", "b"]
    assert all(hit.score < 0 for hit in hits)
    for k in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            retriever.retrieve("word", k)
    retriever.index([make_chunk("new", "z")])
    assert [hit.chunk.id for hit in retriever.retrieve("word", 10)] == ["z"]
