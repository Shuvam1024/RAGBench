from pathlib import Path

import pytest

from ragstat.config import ChunkingConfig
from ragstat.ingestion.chunker import chunk_document
from ragstat.ingestion.loader import Document, load_documents, text_fingerprint


def document(text: str, identifier: str = "source.md") -> Document:
    return Document(id=identifier, text=text, content_sha256=text_fingerprint(text))


@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        ("a b c d e f g h", 5, 2, ["a b c d e", "d e f g h"]),
        ("a b c d e f", 5, 2, ["a b c d e", "d e f"]),
        ("a b c d e", 5, 2, ["a b c d e"]),
        ("a b c d e f", 3, 0, ["a b c", "d e f"]),
        ("a b c", 2, 1, ["a b", "b c"]),
        ("a b", 1, 0, ["a", "b"]),
        ("", 3, 0, []),
        (" \n\t ", 3, 1, []),
        ("  café\n\t東京  hello  ", 2, 0, ["café\n\t東京", "hello"]),
    ],
)
def test_windows(text: str, size: int, overlap: int, expected: list[str]) -> None:
    result = chunk_document(document(text), ChunkingConfig(chunk_size=size, overlap=overlap))
    assert [chunk.text for chunk in result] == expected
    assert all(chunk.end_word - chunk.start_word == len(chunk.text.split()) for chunk in result)


def test_document_unit_indexes_the_full_text_once() -> None:
    text = "a b c d e f"
    source = document(text)
    whole = chunk_document(source, ChunkingConfig(unit="document", chunk_size=2, overlap=0))
    windows = chunk_document(source, ChunkingConfig(chunk_size=2, overlap=0))
    assert [chunk.text for chunk in whole] == [text]
    assert (whole[0].start_word, whole[0].end_word) == (0, 6)
    assert whole[0].id != windows[0].id
    assert chunk_document(document(" \n\t "), ChunkingConfig(unit="document")) == []
    assert whole == chunk_document(source, ChunkingConfig(unit="document"))


def test_chunk_ids_are_stable_and_sensitive_to_inputs() -> None:
    source = document("a b c d e f")
    config = ChunkingConfig(chunk_size=3, overlap=1)
    original = chunk_document(source, config)
    assert original == chunk_document(source, config)
    assert len({chunk.id for chunk in original}) == len(original)
    for changed_source, changed_config in [
        (document("a b c d e z"), config),
        (document(source.text, "renamed.md"), config),
        (source, ChunkingConfig(chunk_size=3, overlap=0)),
    ]:
        assert original[0].id != chunk_document(changed_source, changed_config)[0].id


def test_loader_order_encoding_markdown_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "z.TXT").write_bytes(b"\xef\xbb\xbfline\r\nnext\rend")
    (tmp_path / "a.md").write_text("# café\n\n**bold**", encoding="utf-8")
    (tmp_path / "blank.txt").write_text(" \t\n")
    (tmp_path / "ignored.pdf").write_text("ignore")
    (tmp_path / "link.md").symlink_to(tmp_path / "a.md")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    corpus = load_documents(tmp_path)
    assert [doc.id for doc in corpus.documents] == ["a.md", "nested/z.TXT"]
    assert corpus.documents[0].text == "# café\n\n**bold**"
    assert corpus.documents[1].text == "line\nnext\nend"
    assert corpus.skipped_empty_files == 1


def test_corpus_relocation_preserves_ids(tmp_path: Path) -> None:
    root = tmp_path / "before"
    root.mkdir()
    (root / "a.txt").write_text("alpha beta")
    before = load_documents(root)
    root.rename(tmp_path / "after")
    assert before == load_documents(tmp_path / "after")


def test_jsonl_documents_use_ids_and_skip_blanks(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(
        '\n{"id": "b", "text": "beta"}\n{"id": "a", "text": "  "}\n{"id": "c", "text": "caf\\u00e9"}\n',
        encoding="utf-8",
    )
    corpus = load_documents(tmp_path)
    assert [document.id for document in corpus.documents] == ["b", "c"]
    assert corpus.skipped_empty_files == 1
    assert load_documents(path).documents[1].text == "café"
    path.write_text('{"id": "a", "text": "one"}\n{"id": "a", "text": "two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_documents(path)


def test_loader_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No nonempty"):
        load_documents(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        load_documents(tmp_path / "missing")
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="bad.txt"):
        load_documents(tmp_path)
