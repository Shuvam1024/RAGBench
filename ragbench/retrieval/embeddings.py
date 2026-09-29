"""An injectable embedding boundary and the real Sentence Transformers adapter."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ragbench.config import DenseConfig


class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    @property
    def metadata(self) -> dict[str, str | int]: ...


class SentenceTransformerEmbedder:
    """Resolve a model revision once, encode on CPU, and report truncation."""

    def __init__(self, config: DenseConfig) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from sentence_transformers import SentenceTransformer

        # The small-corpus baseline uses one CPU thread. This also avoids the
        # macOS OpenMP conflict between the PyTorch and FAISS binary runtimes.
        torch.set_num_threads(1)
        config_file = hf_hub_download(config.model_name, "config.json", revision=config.revision)
        self._revision = Path(config_file).parent.name
        self._config = config
        self._model = SentenceTransformer(
            config.model_name,
            revision=self._revision,
            device=config.device,
            trust_remote_code=False,
        )
        self._truncated_texts = 0

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        tokens = self._model.tokenizer(list(texts), truncation=False, padding=False, verbose=False)
        truncated = sum(len(ids) > self._model.max_seq_length for ids in tokens["input_ids"])
        self._truncated_texts += truncated
        if truncated:
            warnings.warn(
                f"{truncated} input text(s) exceed the embedding model's "
                f"{self._model.max_seq_length}-token limit and will be truncated; reduce chunk_size.",
                UserWarning,
                stacklevel=2,
            )
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self._config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "model_name": self._config.model_name,
            "model_revision": self._revision,
            "device": self._config.device,
            "cpu_threads": 1,
            "truncated_texts": self._truncated_texts,
        }
