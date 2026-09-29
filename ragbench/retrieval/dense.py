"""Exact FAISS search over unit-length float32 embeddings."""

from collections.abc import Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from ragbench.config import DenseConfig
from ragbench.ingestion.chunker import Chunk
from ragbench.retrieval.base import Retriever, SearchResult
from ragbench.retrieval.embeddings import EmbeddingProvider, SentenceTransformerEmbedder


def normalize_vectors(vectors: NDArray[np.float32], rows: int) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError("Embeddings must have shape (number of texts, positive dimension)")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must contain only finite values")
    # Float64 norms avoid overflow/underflow for otherwise valid float32 vectors.
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embeddings must not contain zero vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class DenseRetriever(Retriever):
    def __init__(
        self, config: DenseConfig | None = None, embedder: EmbeddingProvider | None = None
    ) -> None:
        super().__init__()
        self.config = config or DenseConfig()
        self._embedder = embedder
        self._index: faiss.IndexFlatIP | None = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        validated = self.validate_chunks(chunks)
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.config)
        vectors = normalize_vectors(
            self._embedder.encode([chunk.text for chunk in validated]), len(validated)
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._chunks, self._index = validated, index

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.validate_query(k)
        if not query.strip():
            return []
        assert self._embedder is not None and self._index is not None
        query_vector = normalize_vectors(self._embedder.encode([query]), 1)
        if query_vector.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension differs from the index")
        # Retrieve all candidates so tied scores at the K boundary are stable.
        scores, positions = self._index.search(query_vector, len(self._chunks))
        aligned = np.empty(len(self._chunks), dtype=np.float32)
        aligned[positions[0]] = scores[0]
        return self.rank(aligned.tolist(), k)

    @property
    def metadata(self) -> dict[str, str | int]:
        return {**super().metadata, **(self._embedder.metadata if self._embedder else {})}
