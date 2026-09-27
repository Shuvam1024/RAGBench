"""Lexical retrieval with identical Unicode tokenization for corpus and query."""

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from ragbench.config import BM25Config
from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.base import Retriever, SearchResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25Retriever(Retriever):
    def __init__(self, config: BM25Config | None = None) -> None:
        super().__init__()
        self.config = config or BM25Config()
        self._index: BM25Okapi | None = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        validated = self.validate_chunks(chunks)
        tokens = [tokenize(chunk.text) for chunk in validated]
        if not any(tokens):
            raise ValueError("BM25 corpus contains no searchable words")
        index = BM25Okapi(tokens, k1=self.config.k1, b=self.config.b)
        self._chunks, self._index = validated, index

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.validate_query(k)
        tokens = tokenize(query)
        if not tokens:
            return []
        assert self._index is not None
        return self.rank(self._index.get_scores(tokens).tolist(), k)
