"""Deterministic word windows with preserved text and stable content-aware IDs."""

import hashlib
import json
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ragbench.config import ChunkingConfig
from ragbench.ingestion.loader import Document


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    document_id: str
    text: str
    start_word: int
    end_word: int


def count_words(text: str) -> int:
    """Count the same non-whitespace spans the chunk windows use."""
    return len(re.findall(r"\S+", text))


def _chunk_id(identity: list[object]) -> str:
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"chunk-v1-{digest}"


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Advance by size minus overlap; stop once a window reaches the final word.

    ``unit: document`` emits the whole document as one chunk. Word-window IDs
    stay on the historical identity so existing chunk fingerprints do not move.
    """
    words = list(re.finditer(r"\S+", document.text))
    if config.unit == "document":
        if not words:
            return []
        return [
            Chunk(
                id=_chunk_id([document.id, document.content_sha256, "document"]),
                document_id=document.id,
                text=document.text,
                start_word=0,
                end_word=len(words),
            )
        ]
    chunks: list[Chunk] = []
    for start in range(0, len(words), config.chunk_size - config.overlap):
        end = min(start + config.chunk_size, len(words))
        chunks.append(
            Chunk(
                id=_chunk_id(
                    [
                        document.id,
                        document.content_sha256,
                        config.chunk_size,
                        config.overlap,
                        start,
                        end,
                    ]
                ),
                document_id=document.id,
                text=document.text[words[start].start() : words[end - 1].end()],
                start_word=start,
                end_word=end,
            )
        )
        if end == len(words):
            break
    return chunks


def chunk_documents(documents: Sequence[Document], config: ChunkingConfig) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, config)]
