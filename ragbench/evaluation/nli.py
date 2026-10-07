"""Pinned three-way NLI scores. Tests can inject a scorer instead of loading weights.

The premise is the evidence sentence and the hypothesis is the claim. Column 0
is contradiction, column 1 is entailment, and column 2 is neutral, matching
``cross-encoder/nli-MiniLM2-L6-H768``. Softmax turns the logits into
probabilities. Entailment maps to SUPPORT, contradiction to CONTRADICT, and
neutral to NEI.
"""

import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from ragbench.config import ConfigModel, Nonblank, PositiveInt
from ragbench.retrieval.rerank import count_overlong


class NliModelConfig(ConfigModel):
    model_name: Nonblank = "cross-encoder/nli-MiniLM2-L6-H768"
    revision: Nonblank = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
    max_length: PositiveInt = 512
    batch_size: PositiveInt = 32
    device: Literal["cpu"] = "cpu"


@dataclass(frozen=True)
class NliScores:
    """Contradiction, entailment, and neutral probabilities."""

    contradiction: float
    entailment: float
    neutral: float

    def __post_init__(self) -> None:
        values = (self.contradiction, self.entailment, self.neutral)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("NLI probabilities must be finite and nonnegative")


class NliScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NliScores]: ...

    @property
    def metadata(self) -> dict[str, str | int]: ...

    @property
    def truncated_pairs(self) -> int: ...


def classify_nli(scores: NliScores) -> tuple[str, float]:
    """Return the verdict and its probability.

    Equal probabilities break toward contradiction, then entailment, then
    neutral. That is the model's label order.
    """
    options = (
        (scores.contradiction, 0, "CONTRADICT"),
        (scores.entailment, 1, "SUPPORT"),
        (scores.neutral, 2, "NEI"),
    )
    probability, _index, label = max(options, key=lambda item: (item[0], -item[1]))
    return label, probability


def load_nli_cross_encoder(config: NliModelConfig) -> tuple[object, str]:
    """Load the pinned NLI cross-encoder on one CPU thread."""
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(1)
    try:
        config_file = hf_hub_download(config.model_name, "config.json", revision=config.revision)
    except Exception as exc:
        raise ValueError(f"Cannot resolve NLI model {config.model_name}: {exc}") from exc
    revision = Path(config_file).parent.name
    try:
        raw = json.loads(Path(config_file).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read NLI config {config_file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("NLI config must be a JSON object")
    labels = raw.get("id2label")
    positional = raw.get("max_position_embeddings")
    expected = {"0": "contradiction", "1": "entailment", "2": "neutral"}
    if (
        not isinstance(labels, dict)
        or {str(key): value for key, value in labels.items()} != expected
    ):
        raise ValueError("NLI model id2label is not contradiction, entailment, neutral")
    if isinstance(positional, bool) or not isinstance(positional, int) or positional <= 0:
        raise ValueError("NLI config did not report max_position_embeddings")
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
        activation_fn=torch.nn.Identity(),
    )
    return model, revision


class CrossEncoderNli:
    """Score (premise, hypothesis) pairs. The premise is a sentence and the hypothesis is the claim."""

    def __init__(
        self, config: NliModelConfig, model: object | None = None, revision: str = ""
    ) -> None:
        self._config = config
        self._truncated_pairs = 0
        self._warned = False
        if model is None:
            model, revision = load_nli_cross_encoder(config)
        if not revision:
            raise ValueError("An injected NLI model needs an explicit revision label")
        self._model = model
        self._revision = revision

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NliScores]:
        if not pairs:
            return []
        import torch

        tokenizer = getattr(self._model, "tokenizer", None)
        predict = getattr(self._model, "predict", None)
        if tokenizer is None or predict is None:
            raise ValueError("NLI model must provide tokenizer and predict")
        encoded = tokenizer(
            [premise for premise, _ in pairs],
            [hypothesis for _, hypothesis in pairs],
            truncation=False,
            padding=False,
            verbose=False,
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        overlong = count_overlong(lengths, self._config.max_length)
        self._truncated_pairs += overlong
        if overlong and not self._warned:
            self._warned = True
            warnings.warn(
                f"{overlong} sentence/claim pair(s) exceed the NLI "
                f"{self._config.max_length}-token limit and will be truncated.",
                UserWarning,
                stacklevel=2,
            )
        raw = predict(
            list(pairs),
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            apply_softmax=True,
            activation_fn=torch.nn.Identity(),
        )
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (len(pairs), 3):
            raise ValueError("Expected one three-class NLI distribution per pair")
        return [NliScores(float(row[0]), float(row[1]), float(row[2])) for row in array]

    @property
    def truncated_pairs(self) -> int:
        return self._truncated_pairs

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "nli_implementation": type(self).__name__,
            "nli_model_name": self._config.model_name,
            "nli_revision": self._revision,
            "nli_max_length": self._config.max_length,
            "nli_cpu_threads": 1,
            "nli_pair_order": "premise_sentence_hypothesis_claim",
            "nli_truncated_pairs": self._truncated_pairs,
        }
