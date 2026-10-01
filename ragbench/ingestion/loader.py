"""Load a portable, deterministically ordered text corpus."""

import hashlib
import json
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


def load_jsonl_documents(path: Path) -> Corpus:
    """Read ``{"id", "text"}`` lines. IDs are the JSON ids, not file names."""
    documents: list[Document] = []
    skipped = 0
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read document {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            document_id = str(row["id"])
            text = str(row["text"]).replace("\r\n", "\n").replace("\r", "\n")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} line {line_number} must contain string id and text") from exc
        if not document_id.strip():
            raise ValueError(f"{path} line {line_number} has an empty id")
        if document_id in seen:
            raise ValueError(f"Duplicate document id {document_id}")
        seen.add(document_id)
        if not text.strip():
            skipped += 1
            continue
        documents.append(Document(id=document_id, text=text, content_sha256=text_fingerprint(text)))
    if not documents:
        raise ValueError(f"No nonempty documents in {path}")
    documents.sort(key=lambda item: item.id)
    return Corpus(documents=tuple(documents), skipped_empty_files=skipped)


def load_documents(root: Path) -> Corpus:
    """Read .txt/.md files, or documents.jsonl when that file is present."""
    root = root.resolve()
    if root.is_file() and root.suffix.lower() == ".jsonl":
        return load_jsonl_documents(root)
    if not root.is_dir():
        raise ValueError(f"Documents directory does not exist: {root}")
    manifest = root / "documents.jsonl"
    if manifest.is_file():
        return load_jsonl_documents(manifest)
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
        documents.append(
            Document(
                id=path.relative_to(root).as_posix(),
                text=text,
                content_sha256=text_fingerprint(text),
            )
        )
    if not documents:
        raise ValueError(f"No nonempty .txt or .md documents found in {root}")
    return Corpus(documents=tuple(documents), skipped_empty_files=skipped)
