"""Load a portable, deterministically ordered text corpus."""

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    text: str
    content_sha256: str


class Corpus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    documents: tuple[Document, ...]
    skipped_empty_files: int = 0


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents(root: Path) -> Corpus:
    """Read .txt/.md files; preserve Markdown, normalize newlines, skip symlinks."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Documents directory does not exist: {root}")
    paths: list[Path] = []

    def fail_walk(error: OSError) -> None:
        raise ValueError(f"Cannot scan documents: {error}") from error

    for folder, directories, filenames in os.walk(root, onerror=fail_walk, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
        for name in filenames:
            path = Path(folder) / name
            if path.suffix.lower() in {".txt", ".md"} and not path.is_symlink():
                paths.append(path)
    documents: list[Document] = []
    skipped = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read document {path}: {exc}") from exc
        if not text.strip():
            skipped += 1
            continue
        documents.append(Document(
            id=path.relative_to(root).as_posix(), text=text, content_sha256=text_fingerprint(text),
        ))
    if not documents:
        raise ValueError(f"No nonempty .txt or .md documents found in {root}")
    return Corpus(documents=tuple(documents), skipped_empty_files=skipped)
