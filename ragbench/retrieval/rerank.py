"""Second-stage cross-encoder scores for a fixed first-stage document set.

The first-stage hit list is already ordered by score. Each document contributes
its first hit, which is the highest-scoring chunk. That chunk's text is the
only string the cross-encoder sees for the document. Later chunks of the same
document are not concatenated and are not scored separately.
"""

import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragbench.config import RerankConfig
from ragbench.retrieval.base import SearchResult


@dataclass(frozen=True)
class ScoredDocument:
    document_id: str
    chunk_id: str
    text: str
    first_stage_score: float


class PairScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...

    @property
    def metadata(self) -> dict[str, str | int]: ...

    @property
    def truncated_pairs(self) -> int: ...


def select_candidates(
    hits: Sequence[SearchResult], candidate_k: int | None
) -> list[ScoredDocument]:
    """Keep the first hit of each document, in the order ``hits`` already has.

    ``candidate_k`` stops after that many distinct documents. ``None`` keeps
    every distinct document. ``hits`` must be the first-stage order: a higher
    score first, with that stage's own tie break already applied.
    """
    if candidate_k is not None and (
        isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0
    ):
        raise ValueError("candidate_k must be a positive integer")
    chosen: list[ScoredDocument] = []
    seen: set[str] = set()
    for hit in hits:
        document_id = hit.chunk.document_id
        if document_id in seen:
            continue
        seen.add(document_id)
        chosen.append(
            ScoredDocument(
                document_id=document_id,
                chunk_id=hit.chunk.id,
                text=hit.chunk.text,
                first_stage_score=hit.score,
            )
        )
        if candidate_k is not None and len(chosen) == candidate_k:
            break
    return chosen


def rank_candidates(
    candidates: Sequence[ScoredDocument], scores: Sequence[float]
) -> list[tuple[ScoredDocument, float]]:
    """Higher cross-encoder score first. Equal scores use the document ID."""
    if len(candidates) != len(scores) or any(isinstance(score, bool) for score in scores):
        raise ValueError("Expected one cross-encoder score per candidate")
    if not all(isinstance(score, int | float) and math.isfinite(float(score)) for score in scores):
        raise ValueError("Expected one finite cross-encoder score per candidate")
    order = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: (-float(pair[1]), pair[0].document_id),
    )
    return [(document, float(score)) for document, score in order]


def count_overlong(lengths: Sequence[int], limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return sum(length > limit for length in lengths)


def load_cross_encoder(config: RerankConfig) -> tuple[object, str]:
    """Load the pinned cross-encoder on one CPU thread."""
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(1)
    try:
        config_file = hf_hub_download(config.model_name, "config.json", revision=config.revision)
    except Exception as exc:
        raise ValueError(f"Cannot resolve cross-encoder {config.model_name}: {exc}") from exc
    revision = Path(config_file).parent.name
    try:
        raw = json.loads(Path(config_file).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read cross-encoder config {config_file}: {exc}") from exc
    positional = raw.get("max_position_embeddings") if isinstance(raw, dict) else None
    if isinstance(positional, bool) or not isinstance(positional, int) or positional <= 0:
        raise ValueError("Cross-encoder config did not report max_position_embeddings")
    if config.max_length > positional:
        raise ValueError(
            f"max_length {config.max_length} exceeds the model positional limit {positional}"
        )
    model = CrossEncoder(
        config.model_name,
        revision=revision,
        device=config.device,
        max_length=config.max_length,
        trust_remote_code=False,
    )
    return model, revision


class CrossEncoderReranker:
    """Score query/document pairs with a sentence-transformers cross-encoder.

    Pairs longer than ``max_length`` tokens, including special tokens, are
    truncated by the tokenizer. The score is the model's raw output. This
    checkpoint's default activation is the identity, so the value is a ranking
    logit rather than a probability.
    """

    def __init__(
        self, config: RerankConfig, model: object | None = None, revision: str = ""
    ) -> None:
        self._config = config
        self._truncated_pairs = 0
        if model is None:
            model, revision = load_cross_encoder(config)
        if not revision:
            raise ValueError("An injected cross-encoder needs an explicit revision label")
        self._model = model
        self._revision = revision

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        import numpy as np

        tokenizer = getattr(self._model, "tokenizer", None)
        predict = getattr(self._model, "predict", None)
        if tokenizer is None or predict is None:
            raise ValueError("Cross-encoder model must provide tokenizer and predict")
        encoded = tokenizer(
            [query for query, _ in pairs],
            [text for _, text in pairs],
            truncation=False,
            padding=False,
            verbose=False,
        )
        input_ids = encoded["input_ids"]
        lengths = [len(ids) for ids in input_ids]
        overlong = count_overlong(lengths, self._config.max_length)
        self._truncated_pairs += overlong
        if overlong:
            warnings.warn(
                f"{overlong} query/document pair(s) exceed the cross-encoder "
                f"{self._config.max_length}-token limit and will be truncated.",
                UserWarning,
                stacklevel=2,
            )
        raw = predict(
            list(pairs),
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = [float(score) for score in np.asarray(raw, dtype=np.float64).reshape(-1)]
        if len(values) != len(pairs) or not all(math.isfinite(value) for value in values):
            raise ValueError("Expected one finite cross-encoder score per pair")
        return values

    @property
    def truncated_pairs(self) -> int:
        return self._truncated_pairs

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "reranker_implementation": type(self).__name__,
            "reranker_model_name": self._config.model_name,
            "reranker_revision": self._revision,
            "reranker_max_length": self._config.max_length,
            "reranker_text": self._config.text,
            "candidate_k": self._config.candidate_k,
            "reranker_cpu_threads": 1,
            "rerank_truncated_pairs": self._truncated_pairs,
        }
