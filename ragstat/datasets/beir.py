"""Checksum-verified BEIR downloads converted into ragstat inputs.

The prepared directory contains ``documents.jsonl`` and ``benchmark.json``.
Document text is the BEIR title and body joined by one newline when both
contain non-whitespace characters. Word lengths use the same ``\\S+`` spans as
the chunker. Relevance grades are written only when a qrel score is not 1;
otherwise the question stays binary. Reference answers are empty because these
files are retrieval labels, not answer keys.
"""

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from statistics import fmean
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from ragstat.evaluation.models import Benchmark
from ragstat.evaluation.timing import percentile
from ragstat.ingestion.chunker import count_words
from ragstat.ingestion.loader import load_documents

SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# SHA-256 of the zip from the UKP BEIR mirror above. NFCorpus is the confirmation
# corpus: its qrels are not used to choose SciFact settings.
NFCORPUS_SHA256 = "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"
NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
TEXT_POLICY = "title and body joined by a newline when both are non-empty; words are \\S+ spans"
LENGTH_THRESHOLDS = (64, 120, 200, 256, 480)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BeirSource(_Record):
    name: str
    urls: tuple[str, ...]
    sha256: str
    split: str = "test"
    splits: tuple[str, ...] = ("test",)

    @model_validator(mode="after")
    def default_split_is_listed(self) -> Self:
        if self.split not in self.splits:
            raise ValueError(f"Default split {self.split!r} must be one of {self.splits}")
        return self


class LengthBucket(_Record):
    words: int
    documents: int


class WordLength(_Record):
    minimum: int
    p50: float
    mean: float
    maximum: int
    longer_than: tuple[LengthBucket, ...]


class GradeCount(_Record):
    grade: int
    qrels: int


class CorpusManifest(_Record):
    dataset: str
    split: str
    source_url: str
    sha256: str
    text_policy: str
    document_count: int
    skipped_empty_documents: int
    query_count: int
    qrel_count: int
    grades: tuple[GradeCount, ...]
    word_length: WordLength


DATASETS: dict[str, BeirSource] = {
    "scifact": BeirSource(
        name="scifact",
        urls=(SCIFACT_URL,),
        sha256=SCIFACT_SHA256,
        splits=("train", "test"),
    ),
    "nfcorpus": BeirSource(
        name="nfcorpus",
        urls=(NFCORPUS_URL,),
        sha256=NFCORPUS_SHA256,
        splits=("train", "dev", "test"),
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}: expected {expected}, got {actual}")


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract a dataset archive and reject absolute paths or parent segments."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for info in package.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe zip member: {info.filename}")
        package.extractall(destination)


def _download(urls: tuple[str, ...], destination: Path, sha256: str) -> None:
    if destination.is_file() and file_sha256(destination) == sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ragstat"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            require_sha256(temporary, sha256)
            temporary.replace(destination)
            return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
            temporary.unlink(missing_ok=True)
    raise ValueError("Could not download a checksum-verified dataset: " + "; ".join(errors))


def _find(root: Path, suffix: str) -> Path:
    matches = [path for path in root.rglob(suffix) if path.is_file() and ".." not in path.parts]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} under {root}, found {len(matches)}")
    return matches[0]


def _identity_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _grade(raw: str) -> int:
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError(f"Relevance grade must be a nonnegative integer, got {raw!r}")
    return int(raw)


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lower().startswith("query-id\t"):
            continue
        parts = raw.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path} line {line_number} must have three tab-separated fields")
        query_id, document_id, score = parts
        grade = _grade(score)
        if grade <= 0:
            continue
        current = rows.setdefault(query_id, {})
        if document_id in current and current[document_id] != grade:
            raise ValueError(f"Conflicting grades for query {query_id} document {document_id}")
        current[document_id] = grade
    if not rows:
        raise ValueError(f"No positive relevance labels in {path}")
    return rows


def _read_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            query_id = str(row["_id"])
            text = str(row["text"]).replace("\r\n", "\n").replace("\r", "\n")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} line {line_number} is not a BEIR query") from exc
        if not query_id.strip() or not text.strip():
            raise ValueError(f"{path} line {line_number} has an empty id or text")
        if query_id in queries:
            raise ValueError(f"Duplicate query id {query_id}")
        queries[query_id] = text
    return queries


def _document_text(title: object, body: object) -> str:
    title_text = str(title or "").replace("\r\n", "\n").replace("\r", "\n")
    body_text = str(body or "").replace("\r\n", "\n").replace("\r", "\n")
    if title_text.strip() and body_text.strip():
        return f"{title_text}\n{body_text}"
    return title_text if title_text.strip() else body_text


def _read_corpus(path: Path) -> tuple[list[tuple[str, str]], int]:
    documents: list[tuple[str, str]] = []
    skipped = 0
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            document_id = str(row["_id"])
            text = _document_text(row.get("title", ""), row.get("text", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} line {line_number} is not a BEIR document") from exc
        if not document_id.strip():
            raise ValueError(f"{path} line {line_number} has an empty id")
        if document_id in seen:
            raise ValueError(f"Duplicate document id {document_id}")
        seen.add(document_id)
        if not text.strip():
            skipped += 1
            continue
        documents.append((document_id, text))
    if not documents:
        raise ValueError(f"No usable documents in {path}")
    return documents, skipped


def artifact_names(split: str, default_split: str) -> tuple[str, str]:
    """Return the benchmark and manifest filenames for a qrels split.

    The dataset's default split keeps the historical ``benchmark.json`` and
    ``manifest.json`` names. Any other split is written beside them so train
    and test can both stay on disk.
    """
    if split == default_split:
        return "benchmark.json", "manifest.json"
    return f"benchmark.{split}.json", f"manifest.{split}.json"


def resolve_split(source: BeirSource, split: str | None) -> str:
    chosen = source.split if split is None else split
    if chosen not in source.splits:
        available = ", ".join(source.splits)
        raise ValueError(
            f"Split {chosen!r} is not available for {source.name}. Available splits: {available}"
        )
    return chosen


def materialize_beir(
    source: Path,
    destination: Path,
    *,
    dataset: str,
    split: str,
    source_url: str,
    sha256: str,
    benchmark_name: str = "benchmark.json",
    manifest_name: str = "manifest.json",
) -> CorpusManifest:
    """Convert an extracted BEIR directory into documents.jsonl and a benchmark."""
    documents, skipped = _read_corpus(_find(source, "corpus.jsonl"))
    queries = _read_queries(_find(source, "queries.jsonl"))
    qrels = _read_qrels(_find(source, f"qrels/{split}.tsv"))
    available = {document_id for document_id, _ in documents}
    questions: list[dict[str, object]] = []
    grade_histogram: dict[int, int] = {}
    qrel_count = 0
    for query_id in sorted(qrels, key=_identity_key):
        if query_id not in queries:
            raise ValueError(f"Query {query_id} has relevance labels but no query text")
        grades = qrels[query_id]
        missing = sorted(set(grades) - available, key=_identity_key)
        if missing:
            raise ValueError(f"Query {query_id} references unknown documents: {missing}")
        ordered = sorted(grades, key=_identity_key)
        question: dict[str, object] = {
            "id": query_id,
            "question": queries[query_id],
            "expected_answer": "",
            "relevant_document_ids": ordered,
        }
        if any(grades[document_id] != 1 for document_id in ordered):
            question["relevance_grades"] = {
                document_id: grades[document_id] for document_id in ordered
            }
        for document_id in ordered:
            grade_histogram[grades[document_id]] = grade_histogram.get(grades[document_id], 0) + 1
            qrel_count += 1
        questions.append(question)
    benchmark = Benchmark.model_validate({"schema_version": 1, "questions": questions})
    destination.mkdir(parents=True, exist_ok=True)
    documents_path = destination / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as handle:
        for document_id, text in sorted(documents, key=lambda item: _identity_key(item[0])):
            handle.write(json.dumps({"id": document_id, "text": text}, ensure_ascii=False) + "\n")
    (destination / benchmark_name).write_text(
        benchmark.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )
    loaded = load_documents(destination)
    loaded_ids = {document.id for document in loaded.documents}
    for question_model in benchmark.questions:
        missing_ids = set(question_model.relevant_document_ids) - loaded_ids
        if missing_ids:
            raise ValueError(
                f"Prepared benchmark references missing documents: {sorted(missing_ids)}"
            )
    lengths = [count_words(text) for _, text in documents]
    manifest = CorpusManifest(
        dataset=dataset,
        split=split,
        source_url=source_url,
        sha256=sha256,
        text_policy=TEXT_POLICY,
        document_count=len(documents),
        skipped_empty_documents=skipped,
        query_count=len(questions),
        qrel_count=qrel_count,
        grades=tuple(
            GradeCount(grade=grade, qrels=grade_histogram[grade])
            for grade in sorted(grade_histogram)
        ),
        word_length=WordLength(
            minimum=min(lengths),
            p50=percentile(lengths, 0.5),
            mean=fmean(lengths),
            maximum=max(lengths),
            longer_than=tuple(
                LengthBucket(
                    words=threshold, documents=sum(length > threshold for length in lengths)
                )
                for threshold in LENGTH_THRESHOLDS
            ),
        ),
    )
    (destination / manifest_name).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_dataset(
    name: str,
    cache: Path,
    *,
    manifest_path: Path | None = None,
    force: bool = False,
    split: str | None = None,
) -> CorpusManifest:
    """Download, verify, and materialize a known BEIR dataset into ``cache/name``.

    ``split`` selects the qrels file. Omitting it keeps the dataset's default
    split, which is ``test``, and still writes ``benchmark.json``.
    """
    try:
        source = DATASETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}. Known datasets: {known}") from exc
    chosen = resolve_split(source, split)
    benchmark_name, manifest_name = artifact_names(chosen, source.split)
    root = cache.resolve() / source.name
    archive = root / f"{source.name}.zip"
    extracted = root / "raw"
    prepared = root
    manifest_file = prepared / manifest_name
    ready = (
        not force
        and archive.is_file()
        and file_sha256(archive) == source.sha256
        and (prepared / "documents.jsonl").is_file()
        and (prepared / benchmark_name).is_file()
        and manifest_file.is_file()
    )
    if ready:
        manifest = CorpusManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        if manifest.split != chosen:
            raise ValueError(
                f"{manifest_file.name} records split {manifest.split!r}, expected {chosen!r}"
            )
    else:
        _download(source.urls, archive, source.sha256)
        if extracted.exists():
            shutil.rmtree(extracted)
        extract_zip(archive, extracted)
        manifest = materialize_beir(
            extracted,
            prepared,
            dataset=source.name,
            split=chosen,
            source_url=source.urls[0],
            sha256=source.sha256,
            benchmark_name=benchmark_name,
            manifest_name=manifest_name,
        )
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest
