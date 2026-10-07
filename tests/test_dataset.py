import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pytest

from ragbench.datasets.beir import (
    DATASETS,
    _download,
    artifact_names,
    extract_zip,
    file_sha256,
    materialize_beir,
    prepare_dataset,
    require_sha256,
    resolve_split,
)
from ragbench.ingestion.loader import load_documents


def _source(root: Path, *, grade: str = "1") -> None:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "10", "title": "Alpha title", "text": "alpha body " * 30}),
                json.dumps({"_id": "2", "title": "", "text": "beta only"}),
                json.dumps({"_id": "3", "title": "   ", "text": "   "}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "2", "text": "second claim"}),
                json.dumps({"_id": "10", "text": "first claim"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n10\t2\t" + grade + "\n10\t10\t1\n2\t2\t1\n",
        encoding="utf-8",
    )


def test_materialize_orders_ids_and_keeps_binary_labels(tmp_path: Path) -> None:
    source, destination = tmp_path / "raw", tmp_path / "out"
    _source(source)
    manifest = materialize_beir(
        source,
        destination,
        dataset="toy",
        split="test",
        source_url="https://example.test/toy.zip",
        sha256="abc",
    )
    documents = load_documents(destination)
    assert [document.id for document in documents.documents] == ["10", "2"]
    assert documents.documents[0].text.startswith("Alpha title\n")
    assert manifest.skipped_empty_documents == 1
    assert documents.skipped_empty_files == 0
    benchmark = json.loads((destination / "benchmark.json").read_text(encoding="utf-8"))
    assert [question["id"] for question in benchmark["questions"]] == ["2", "10"]
    assert "relevance_grades" not in benchmark["questions"][0]
    assert benchmark["questions"][0]["expected_answer"] == ""
    assert manifest.query_count == 2
    assert manifest.document_count == 2
    assert manifest.word_length.longer_than[0].words == 64


def test_materialize_writes_nonbinary_grades(tmp_path: Path) -> None:
    source, destination = tmp_path / "raw", tmp_path / "out"
    _source(source, grade="2")
    materialize_beir(
        source,
        destination,
        dataset="toy",
        split="test",
        source_url="https://example.test/x",
        sha256="abc",
    )
    question = json.loads((destination / "benchmark.json").read_text(encoding="utf-8"))[
        "questions"
    ][1]
    assert question["id"] == "10"
    assert question["relevance_grades"] == {"2": 2, "10": 1}


def test_checksum_and_zip_slip(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"ragbench")
    digest = file_sha256(payload)
    require_sha256(payload, digest)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        require_sha256(payload, "0" * 64)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError, match="Unsafe zip member"):
        extract_zip(archive, tmp_path / "out")


def test_download_skips_a_matching_file_and_rejects_a_bad_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "ready.zip"
    payload.write_bytes(b"already-correct")
    digest = file_sha256(payload)

    def fail_open(*args: object, **kwargs: object) -> io.BytesIO:
        raise AssertionError("download should not run when the checksum already matches")

    monkeypatch.setattr(urllib.request, "urlopen", fail_open)
    _download(("https://example.test/ready.zip",), payload, digest)

    def bad_open(*args: object, **kwargs: object) -> io.BytesIO:
        return io.BytesIO(b"wrong-bytes")

    monkeypatch.setattr(urllib.request, "urlopen", bad_open)
    with pytest.raises(ValueError, match="checksum-verified"):
        _download(("https://example.test/bad.zip",), tmp_path / "bad.zip", digest)


def test_train_split_is_written_beside_the_default_benchmark(tmp_path: Path) -> None:
    source, destination = tmp_path / "raw", tmp_path / "out"
    _source(source)
    (source / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n2\t10\t1\n",
        encoding="utf-8",
    )
    test_manifest = materialize_beir(
        source,
        destination,
        dataset="toy",
        split="test",
        source_url="https://example.test/toy.zip",
        sha256="abc",
    )
    train_manifest = materialize_beir(
        source,
        destination,
        dataset="toy",
        split="train",
        source_url="https://example.test/toy.zip",
        sha256="abc",
        benchmark_name="benchmark.train.json",
        manifest_name="manifest.train.json",
    )
    assert test_manifest.split == "test" and test_manifest.query_count == 2
    assert train_manifest.split == "train" and train_manifest.query_count == 1
    assert (destination / "benchmark.json").is_file()
    assert (destination / "benchmark.train.json").is_file()
    train_ids = [
        question["id"]
        for question in json.loads((destination / "benchmark.train.json").read_text())["questions"]
    ]
    assert train_ids == ["2"]
    assert artifact_names("test", "test") == ("benchmark.json", "manifest.json")
    assert artifact_names("train", "test") == ("benchmark.train.json", "manifest.train.json")
    assert resolve_split(DATASETS["scifact"], None) == "test"
    assert resolve_split(DATASETS["scifact"], "train") == "train"
    with pytest.raises(ValueError, match="not available"):
        resolve_split(DATASETS["scifact"], "dev")
    nfcorpus = DATASETS["nfcorpus"]
    assert nfcorpus.sha256 == "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"
    assert nfcorpus.splits == ("train", "dev", "test")
    assert resolve_split(nfcorpus, None) == "test"
    assert resolve_split(nfcorpus, "dev") == "dev"


def test_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        prepare_dataset("missing", tmp_path)
    from typer.testing import CliRunner

    from ragbench.cli import app

    result = CliRunner().invoke(app, ["dataset", "missing", "--cache", str(tmp_path)])
    assert result.exit_code == 1
    assert "Unknown dataset" in result.output
    rejected = CliRunner().invoke(
        app, ["dataset", "scifact", "--split", "dev", "--cache", str(tmp_path)]
    )
    assert rejected.exit_code == 1
    assert "not available" in rejected.output
