"""Lexical retrieval with identical tokenization for corpus and query.

``whitespace`` keeps the historical ``\\w+`` tokenizer. ``stem`` uses that same
split, drops a fixed English stopword list, and applies the Snowball English
stemmer from the ``snowballstemmer`` package.
"""

import re
from collections.abc import Sequence

import snowballstemmer
from rank_bm25 import BM25Okapi

from ragbench.config import BM25Config
from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.base import Retriever, SearchResult
from ragbench.retrieval.stopwords import ENGLISH_STOPWORDS

_STEMMER = snowballstemmer.stemmer("english")


def tokenize(text: str, tokenizer: str = "whitespace") -> list[str]:
    """Tokenize ``text`` the same way for indexing and querying."""
    tokens = re.findall(r"\w+", text.casefold())
    if tokenizer == "whitespace":
        return tokens
    if tokenizer == "stem":
        kept = [token for token in tokens if token not in ENGLISH_STOPWORDS]
        return list(_STEMMER.stemWords(kept)) if kept else []
    raise ValueError(f"Unknown BM25 tokenizer: {tokenizer}")


class BM25Retriever(Retriever):
    def __init__(self, config: BM25Config | None = None) -> None:
        super().__init__()
        self.config = config or BM25Config()
        self._index: BM25Okapi | None = None

    @property
    def metadata(self) -> dict[str, str | int]:
        return {**super().metadata, "tokenizer": self.config.tokenizer}

    def index(self, chunks: Sequence[Chunk]) -> None:
        validated = self.validate_chunks(chunks)
        tokens = [tokenize(chunk.text, self.config.tokenizer) for chunk in validated]
        if not any(tokens):
            raise ValueError("BM25 corpus contains no searchable words")
        index = BM25Okapi(tokens, k1=self.config.k1, b=self.config.b)
        self._chunks, self._index = validated, index

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.validate_query(k)
        tokens = tokenize(query, self.config.tokenizer)
        if not tokens:
            return []
        assert self._index is not None
        return self.rank(self._index.get_scores(tokens).tolist(), k)
