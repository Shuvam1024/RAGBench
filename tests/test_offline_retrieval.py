"""Offline coverage for dense search, embeddings, NLI, and reranking.

These tests inject fakes. They do not download weights or import FAISS.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from ragstat import __version__
from ragstat.config import DenseConfig, RerankConfig
from ragstat.evaluation.nli import (
    CrossEncoderNli,
    NliModelConfig,
    NliScores,
    _row_softmax,
    classify_nli,
)
from ragstat.ingestion.chunker import Chunk
from ragstat.retrieval.dense import DenseRetriever, normalize_vectors
from ragstat.retrieval.embeddings import SentenceTransformerEmbedder
from ragstat.retrieval.rerank import CrossEncoderReranker, count_overlong, rank_candidates


class _NumpyIndex:
    def __init__(self, vectors: NDArray[np.float32]) -> None:
        self._vectors = vectors
        self.d = int(vectors.shape[1])

    def search(
        self, query: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        scores = self._vectors @ query[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return scores[order][None, :].astype(np.float32), order[None, :].astype(np.int64)


class _Vectors:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    @property
    def metadata(self) -> dict[str, str | int]:
        return {"model_name": "fixed"}


class _Tokenizer:
    def __call__(
        self,
        left: Sequence[str],
        right: Sequence[str] | None = None,
        truncation: bool = False,
        padding: bool = False,
        verbose: bool = False,
    ) -> dict[str, list[list[int]]]:
        del truncation, padding, verbose
        if right is None:
            return {"input_ids": [[1] * max(len(text), 1) for text in left]}
        return {
            "input_ids": [
                [1] * (len(query) + len(text)) for query, text in zip(left, right, strict=True)
            ]
        }


class _EmbedModel:
    max_seq_length = 4

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.calls = 0

    def encode(self, texts: Sequence[str], **_kwargs: object) -> NDArray[np.float32]:
        self.calls += 1
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


class _PredictModel:
    def __init__(self, rows: list[list[float]] | list[float]) -> None:
        self.tokenizer = _Tokenizer()
        self.rows = rows

    def predict(self, pairs: Sequence[tuple[str, str]], **_kwargs: object) -> object:
        return self.rows


def test_package_version_comes_from_metadata() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_normalize_vectors_rejects_bad_matrices() -> None:
    good = normalize_vectors(np.asarray([[3.0, 4.0]], dtype=np.float32), 1)
    assert good[0].tolist() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError, match="shape"):
        normalize_vectors(np.asarray([[1.0, 0.0]], dtype=np.float32), 2)
    with pytest.raises(ValueError, match="finite"):
        normalize_vectors(np.asarray([[np.nan, 1.0]], dtype=np.float32), 1)
    with pytest.raises(ValueError, match="zero"):
        normalize_vectors(np.asarray([[0.0, 0.0]], dtype=np.float32), 1)


def test_dense_retriever_uses_an_injected_index(
    make_chunk: Callable[..., Chunk],
) -> None:
    embedder = _Vectors({"east": [3, 0], "north": [0, 4], "query": [1, 3]})
    retriever = DenseRetriever(embedder=embedder, index_factory=_NumpyIndex)
    retriever.index([make_chunk("east", "e"), make_chunk("north", "n")])
    hits = retriever.retrieve("query", 2)
    assert [hit.chunk.id for hit in hits] == ["n", "e"]
    assert retriever.retrieve("  ", 1) == []
    embedder.vectors["query"] = [1.0]
    with pytest.raises(ValueError, match="dimension"):
        retriever.retrieve("query", 1)


def test_injected_embedder_counts_truncation_and_caches() -> None:
    model = _EmbedModel()
    embedder = SentenceTransformerEmbedder(DenseConfig(), model=model, revision="test-revision")
    with pytest.warns(UserWarning, match="truncated"):
        first = embedder.encode(["abcd", "toolong"])
    assert embedder.metadata["truncated_texts"] == 1
    with pytest.warns(UserWarning, match="truncated"):
        second = embedder.encode(["abcd", "toolong"])
    assert second is first
    assert model.calls == 1
    alone = embedder.encode(["ab"])
    assert alone.shape == (1, 2)
    assert model.calls == 2
    with pytest.raises(ValueError, match="revision"):
        SentenceTransformerEmbedder(model=model, revision="")


def test_injected_nli_softmax_and_bad_output() -> None:
    assert classify_nli(NliScores(0.2, 0.2, 0.2)) == ("CONTRADICT", 0.2)
    softened = _row_softmax(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64))
    assert softened[0].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    scorer = CrossEncoderNli(
        NliModelConfig(max_length=3),
        model=_PredictModel([[0.0, 2.0, 0.0]]),
        revision="test-revision",
    )
    with pytest.warns(UserWarning, match="truncated"):
        scores = scorer.score([("premise", "hypothesis is long")])
    assert scores[0].entailment > scores[0].contradiction
    assert scorer.truncated_pairs > 0
    assert scorer.score([]) == []
    broken = CrossEncoderNli(
        NliModelConfig(), model=_PredictModel([[1.0, 0.0]]), revision="test-revision"
    )
    with pytest.raises(ValueError, match="three-class"):
        broken.score([("a", "b")])
    with pytest.raises(ValueError, match="revision"):
        CrossEncoderNli(NliModelConfig(), model=_PredictModel([]), revision="")


def test_injected_reranker_warns_and_rejects_bad_scores() -> None:
    assert count_overlong([1, 5], 4) == 1
    with pytest.raises(ValueError, match="positive"):
        count_overlong([1], 0)
    reranker = CrossEncoderReranker(
        RerankConfig(max_length=3),
        model=_PredictModel([0.5]),
        revision="test-revision",
    )
    with pytest.warns(UserWarning, match="truncated"):
        assert reranker.score([("query", "document text")]) == pytest.approx([0.5])
    assert reranker.metadata["rerank_truncated_pairs"] == 1
    assert reranker.score([]) == []
    empty_model = _PredictModel([])
    bad = CrossEncoderReranker(RerankConfig(), model=empty_model, revision="test-revision")
    with pytest.raises(ValueError, match="one finite"):
        bad.score([("q", "d")])
    ranked = rank_candidates([], [])
    assert ranked == []
