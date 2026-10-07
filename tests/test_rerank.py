"""Cross-encoder candidate selection without downloading model weights."""

import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ragstat.cli import app
from ragstat.config import RerankConfig, load_config
from ragstat.evaluation.runner import evaluate, evaluate_candidate_depths
from ragstat.evaluation.train_selection import choose_candidate_k
from ragstat.ingestion.chunker import Chunk
from ragstat.retrieval.base import Retriever, SearchResult
from ragstat.retrieval.rerank import (
    CrossEncoderReranker,
    count_overlong,
    rank_candidates,
    select_candidates,
)


class _RecordingScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.batches: list[list[str]] = []
        self._truncated = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        texts = [text for _, text in pairs]
        self.batches.append(texts)
        return [self.scores[text] for text in texts]

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "reranker_implementation": "recording",
            "reranker_model_name": "recording",
            "reranker_revision": "test-revision",
            "reranker_text": "best_first_stage_chunk",
            "candidate_k": 2,
            "rerank_truncated_pairs": self._truncated,
        }

    @property
    def truncated_pairs(self) -> int:
        return self._truncated


class _OrderedRetriever(Retriever):
    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = self.validate_chunks(chunks)

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.validate_query(k)
        # Higher score first. Document a has two chunks; the alpha chunk wins.
        preference = {"alpha alpha": 5.0, "beta beta": 1.0, "other document": 4.0, "here now": 0.5}
        return self.rank([preference[chunk.text] for chunk in self._chunks], k)


def _experiment(tmp_path: Path, candidate_k: int = 2) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha alpha beta beta")
    (docs / "b.txt").write_text("other document here now")
    benchmark = {
        "schema_version": 1,
        "questions": [
            {
                "id": "q1",
                "question": "alpha",
                "expected_answer": "alpha",
                "relevant_document_ids": ["a.txt"],
            }
        ],
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark))
    config = tmp_path / "run.yaml"
    config.write_text(
        "dataset:\n  documents_path: docs\n  benchmark_path: benchmark.json\n"
        "chunking:\n  chunk_size: 2\n  overlap: 0\n"
        "evaluation:\n  recall_at_k: [1, 100]\n  ndcg_at_k: [10]\n  stored_hits: 10\n"
        "rerank:\n"
        f"  candidate_k: {candidate_k}\n"
        "  max_length: 32\n"
        "  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2\n"
        "  revision: 233902d25c440f23af6f7d6e94d2946bac0bee0a\n"
        "  text: best_first_stage_chunk\n"
    )
    return config


def test_best_chunk_is_the_only_cross_encoder_text(make_chunk: Callable[..., Chunk]) -> None:
    hits = [
        SearchResult(chunk=make_chunk("best", "c2", document_id="a"), score=3.0),
        SearchResult(chunk=make_chunk("later", "c1", document_id="a"), score=2.0),
        SearchResult(chunk=make_chunk("other", "c3", document_id="b"), score=1.0),
    ]
    chosen = select_candidates(hits, 2)
    assert [(item.document_id, item.text) for item in chosen] == [("a", "best"), ("b", "other")]
    ranked = rank_candidates(chosen, [0.1, 0.1])
    assert [item.document_id for item, _score in ranked] == ["a", "b"]
    ranked = rank_candidates(chosen, [0.2, 5.0])
    assert [item.document_id for item, _score in ranked] == ["b", "a"]
    with pytest.raises(ValueError, match="one cross-encoder score"):
        rank_candidates(chosen, [0.1])
    with pytest.raises(ValueError, match="finite"):
        rank_candidates(chosen, [0.1, float("inf")])
    with pytest.raises(ValueError, match="positive"):
        select_candidates(hits, 0)
    with pytest.raises(ValueError, match="positive"):
        count_overlong([1], True)
    assert count_overlong([8, 9, 4], 8) == 1


def test_rerank_reorders_candidates_and_keeps_first_stage_recall(tmp_path: Path) -> None:
    config = load_config(_experiment(tmp_path))
    scorer = _RecordingScorer({"alpha alpha": 0.0, "other document": 4.0})
    result = evaluate(config, retriever=_OrderedRetriever(), reranker=scorer)
    question = result.questions[0]
    assert scorer.batches == [["alpha alpha", "other document"]]
    assert [hit.document_id for hit in question.retrieved_documents] == ["b.txt", "a.txt"]
    assert question.reciprocal_rank == 0.5
    assert question.candidate_recall_at_100 == 1
    assert question.recall_at_k[100] == 1
    assert question.pipeline_ms == pytest.approx(question.retrieval_ms + question.rerank_ms)
    assert result.candidate_recall_at_100 == 1
    assert result.timings.rerank_p95_ms is not None
    assert result.timings.pipeline_p95_ms is not None
    assert result.retriever["reranker_text"] == "best_first_stage_chunk"
    assert result.retriever["candidate_k"] == 2
    assert result.config.rerank is not None
    assert result.config.rerank.candidate_k == 2


def test_candidate_depth_grid_uses_prefixes_of_one_scoring_pass(tmp_path: Path) -> None:
    config = load_config(_experiment(tmp_path, candidate_k=2))
    scorer = _RecordingScorer({"alpha alpha": 0.0, "other document": 4.0})
    reports = evaluate_candidate_depths(
        config, (1, 2), retriever=_OrderedRetriever(), reranker=scorer
    )
    assert scorer.batches == [["alpha alpha", "other document"]]
    assert reports[1].questions[0].reciprocal_rank == 1
    assert reports[1].questions[0].rerank_ms is None
    assert reports[2].questions[0].rerank_ms is not None
    assert reports[2].questions[0].reciprocal_rank == 0.5
    assert reports[1].candidate_recall_at_100 == reports[2].candidate_recall_at_100 == 1
    assert reports[1].config.rerank is not None and reports[1].config.rerank.candidate_k == 1
    winner = choose_candidate_k(
        [
            (depth, report.ndcg_at_k[10])
            for depth, report in reports.items()
            if report.ndcg_at_k is not None
        ]
    )
    assert winner == 1
    with pytest.raises(ValueError, match="widest"):
        evaluate_candidate_depths(config, (1,), retriever=_OrderedRetriever(), reranker=scorer)


def test_injected_cross_encoder_counts_truncation_and_rejects_bad_scores() -> None:
    class _Tokenizer:
        def __call__(
            self,
            queries: Sequence[str],
            texts: Sequence[str],
            truncation: bool,
            padding: bool,
            verbose: bool,
        ) -> dict[str, list[list[int]]]:
            assert truncation is False
            return {"input_ids": [[1] * (4 if text == "short" else 12) for text in texts]}

    class _Model:
        def __init__(self, values: list[float]) -> None:
            self.tokenizer = _Tokenizer()
            self.values = values

        def predict(
            self,
            pairs: Sequence[tuple[str, str]],
            batch_size: int,
            show_progress_bar: bool,
            convert_to_numpy: bool,
        ) -> np.ndarray:
            assert batch_size == 4
            assert show_progress_bar is False
            if len(pairs) == len(self.values):
                return np.asarray(self.values, dtype=np.float64)
            return np.full(len(pairs), self.values[-1], dtype=np.float64)

    config = RerankConfig(max_length=8, batch_size=4, revision="abc")
    reranker = CrossEncoderReranker(config, model=_Model([0.2, 1.5]), revision="abc")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert reranker.score([("q", "short"), ("q", "long")]) == [0.2, 1.5]
    assert reranker.truncated_pairs == 1
    assert any("truncated" in str(item.message) for item in caught)
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        assert reranker.score([("q", "long")]) == [1.5]
    assert reranker.truncated_pairs == 2
    assert again == []
    broken = CrossEncoderReranker(config, model=_Model([float("nan")]), revision="abc")
    with pytest.raises(ValueError, match="finite"):
        broken.score([("q", "short")])
    assert reranker.score(()) == []
    assert reranker.metadata["reranker_revision"] == "abc"
    with pytest.raises(ValueError, match="revision"):
        CrossEncoderReranker(config, model=_Model([0.0]), revision="")
    with pytest.raises(ValueError, match="tokenizer"):
        CrossEncoderReranker(config, model=object(), revision="abc").score([("q", "short")])


def test_cli_labels_the_reranked_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _experiment(tmp_path)
    prepared = evaluate(
        load_config(config),
        retriever=_OrderedRetriever(),
        reranker=_RecordingScorer({"alpha alpha": 0.0, "other document": 4.0}),
    )
    monkeypatch.setattr("ragstat.cli.run_evaluation", lambda settings: prepared)
    result = CliRunner().invoke(
        app, ["evaluate", "--config", str(config), "--output", str(tmp_path / "out.json")]
    )
    assert result.exit_code == 0, result.output
    assert "Candidate Recall@100 1.0000 (first stage, before rerank)" in result.output
    assert "MRR        0.5000 (reranked candidate set)" in result.output
    assert "Full-ranking retrieval p95:" in result.output
    assert "Rerank p95:" in result.output
    assert "Top-K pipeline p95:" in result.output


def test_train_rerank_config_keeps_the_frozen_hybrid() -> None:
    config = load_config(Path("configs/scifact-rerank-train.yaml"))
    assert config.rerank is not None
    assert config.rerank.candidate_k == 100
    assert config.rerank.max_length == 512
    assert config.rerank.text == "best_first_stage_chunk"
    assert config.rerank.revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert config.dataset.benchmark_path.name == "benchmark.train.json"
    assert config.chunking.chunk_size == 480
    assert config.retrieval.type == "hybrid"


def test_rerank_settings_reject_unknown_text() -> None:
    with pytest.raises(ValueError):
        RerankConfig.model_validate({"text": "full_document", "candidate_k": 10})
    with pytest.raises(ValueError, match="reranker requires"):
        evaluate(
            load_config(Path("configs/baseline.yaml")),
            reranker=_RecordingScorer({}),
        )
