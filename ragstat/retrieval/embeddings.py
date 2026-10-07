"""An injectable embedding boundary and the real Sentence Transformers adapter."""

import hashlib
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ragstat.config import DenseConfig


class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    @property
    def metadata(self) -> dict[str, str | int]: ...


_ENCODE_CACHE: dict[tuple[str, str, str], NDArray[np.float32]] = {}


class SentenceTransformerEmbedder:
    """Resolve a model revision once, encode on CPU, and report truncation."""

    def __init__(
        self,
        config: DenseConfig | None = None,
        model: object | None = None,
        revision: str = "",
    ) -> None:
        self._config = config or DenseConfig()
        self._truncated_texts = 0
        self._model: Any
        if model is None:
            import torch
            from huggingface_hub import hf_hub_download
            from sentence_transformers import SentenceTransformer

            # The small-corpus baseline uses one CPU thread. This also avoids the
            # macOS OpenMP conflict between the PyTorch and FAISS binary runtimes.
            torch.set_num_threads(1)
            config_file = hf_hub_download(
                self._config.model_name, "config.json", revision=self._config.revision
            )
            self._revision = Path(config_file).parent.name
            self._model = SentenceTransformer(
                self._config.model_name,
                revision=self._revision,
                device=self._config.device,
                trust_remote_code=False,
            )
        else:
            if not revision:
                raise ValueError("An injected embedding model needs an explicit revision label")
            self._model = model
            self._revision = revision

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        tokens = self._model.tokenizer(list(texts), truncation=False, padding=False, verbose=False)
        limit = self._model.max_seq_length
        if not isinstance(limit, int):
            raise ValueError("Embedding model did not report an integer sequence limit")
        truncated = sum(len(ids) > limit for ids in tokens["input_ids"])
        self._truncated_texts += truncated
        if truncated:
            warnings.warn(
                f"{truncated} input text(s) exceed the embedding model's "
                f"{self._model.max_seq_length}-token limit and will be truncated; reduce chunk_size.",
                UserWarning,
                stacklevel=2,
            )
        digest = hashlib.sha256()
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        key = (self._config.model_name, self._revision, digest.hexdigest())
        cached = _ENCODE_CACHE.get(key)
        if cached is not None and len(cached) == len(texts):
            return cached
        vectors = np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self._config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        if len(texts) > 1:
            _ENCODE_CACHE[key] = vectors
        return vectors

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "model_name": self._config.model_name,
            "model_revision": self._revision,
            "device": self._config.device,
            "cpu_threads": 1,
            "truncated_texts": self._truncated_texts,
        }
