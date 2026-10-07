"""Original SciFact claims, abstracts, and the BEIR query-id mapping.

Verdicts and evidence sentence IDs come from the AllenAI claim files described
in https://github.com/allenai/scifact/blob/master/doc/data.md. BEIR qrels are
retrieval grades. This module does not read a verdict out of a qrel.
"""

import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ragstat.config import ConfigModel, Nonblank
from ragstat.datasets.beir import _download, file_sha256

SCIFACT_CLAIMS_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_CLAIMS_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
ClaimSplit = Literal["train", "dev", "test"]
_ARCHIVE_MEMBERS = {
    "data/claims_train.jsonl": "claims_train.jsonl",
    "data/claims_dev.jsonl": "claims_dev.jsonl",
    "data/claims_test.jsonl": "claims_test.jsonl",
    "data/corpus.jsonl": "corpus.jsonl",
}


class EvidenceSentence(ConfigModel):
    document_id: Nonblank
    sentence: int = Field(ge=0)
    label: Literal["SUPPORT", "CONTRADICT"]


class SciFactClaim(ConfigModel):
    """One claim. ``labels_withheld`` is the public test file, which has no evidence."""

    id: Nonblank
    text: Nonblank
    evidence: tuple[EvidenceSentence, ...] = ()
    cited_doc_ids: tuple[Nonblank, ...] = ()
    labels_withheld: bool = False

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> "SciFactClaim":
        if self.labels_withheld and (self.evidence or self.cited_doc_ids):
            raise ValueError(f"Claim {self.id} cannot withhold labels and carry evidence")
        seen: set[tuple[str, int]] = set()
        labels = {item.label for item in self.evidence}
        if len(labels) > 1:
            raise ValueError(f"Claim {self.id} has both SUPPORT and CONTRADICT rationales")
        for item in self.evidence:
            key = (item.document_id, item.sentence)
            if key in seen:
                raise ValueError(f"Claim {self.id} repeats evidence sentence {key}")
            seen.add(key)
        return self

    def gold_verdict(self) -> Literal["SUPPORT", "CONTRADICT", "NEI"]:
        """NEI when labeled evidence is empty. The public test file has no gold label."""
        if self.labels_withheld:
            raise ValueError(
                f"Claim {self.id} is in the public test split and its label is withheld"
            )
        labels = {item.label for item in self.evidence}
        if not labels:
            return "NEI"
        if "SUPPORT" in labels:
            return "SUPPORT"
        return "CONTRADICT"

    def gold_sentences(self) -> frozenset[tuple[str, int]]:
        """``(document id, abstract sentence index)`` pairs. Not title tokens."""
        if self.labels_withheld:
            raise ValueError(
                f"Claim {self.id} is in the public test split and its label is withheld"
            )
        return frozenset((item.document_id, item.sentence) for item in self.evidence)

    def evidence_document_ids(self) -> frozenset[str]:
        return frozenset(item.document_id for item in self.evidence)


class AbstractDocument(ConfigModel):
    id: Nonblank
    title: str
    sentences: tuple[Nonblank, ...]

    @model_validator(mode="after")
    def abstract_has_a_sentence(self) -> "AbstractDocument":
        if not self.sentences:
            raise ValueError(f"Document {self.id} has no abstract sentences")
        return self


class SplitAlignment(ConfigModel):
    """How one claim split lines up with BEIR queries and, when present, qrels."""

    claim_split: str
    beir_qrel_split: str | None
    claim_count: int = Field(ge=0)
    query_text_matches: int = Field(ge=0)
    claims_missing_from_queries: int = Field(ge=0)
    qrel_equals_cited: int = Field(ge=0)
    qrel_equals_evidence: int = Field(ge=0)
    evidence_subset_of_cited: int = Field(ge=0)
    grades: tuple[int, ...] = ()


class MappingReport(ConfigModel):
    train: SplitAlignment
    dev: SplitAlignment
    official_test: SplitAlignment
    query_count: int = Field(ge=0)
    whitespace_normalized_body_mismatches: int | None = None
    exact_space_join_mismatches: int | None = None


def _identifier(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return str(value)


def _parse_evidence(claim_id: str, raw: object) -> tuple[EvidenceSentence, ...]:
    if not isinstance(raw, dict):
        raise ValueError(f"Claim {claim_id} evidence must be an object")
    sentences: list[EvidenceSentence] = []
    for document_id, rationales in raw.items():
        if not isinstance(rationales, list):
            raise ValueError(f"Claim {claim_id} evidence for {document_id} must be a list")
        for rationale in rationales:
            if not isinstance(rationale, dict):
                raise ValueError(f"Claim {claim_id} has a malformed rationale")
            label = rationale.get("label")
            indexes = rationale.get("sentences")
            if (
                label not in ("SUPPORT", "CONTRADICT")
                or not isinstance(indexes, list)
                or not indexes
            ):
                raise ValueError(
                    f"Claim {claim_id} has a rationale without SUPPORT/CONTRADICT sentences"
                )
            for index in indexes:
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError(f"Claim {claim_id} has a bad sentence index")
                sentences.append(
                    EvidenceSentence(document_id=str(document_id), sentence=index, label=label)
                )
    return tuple(sentences)


def parse_claim(row: object, *, split: ClaimSplit) -> SciFactClaim:
    """Parse one JSON object from ``claims_{split}.jsonl``."""
    if not isinstance(row, dict):
        raise ValueError("A SciFact claim must be a JSON object")
    claim_id = _identifier(row.get("id"), label="Claim id")
    text = row.get("claim")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Claim {claim_id} has no claim text")
    if split == "test":
        if "evidence" in row or "cited_doc_ids" in row:
            raise ValueError(f"Public test claim {claim_id} is not expected to carry labels")
        return SciFactClaim(id=claim_id, text=text, labels_withheld=True)
    if "evidence" not in row or "cited_doc_ids" not in row:
        raise ValueError(f"Claim {claim_id} is missing evidence or cited_doc_ids")
    cited_raw = row["cited_doc_ids"]
    if not isinstance(cited_raw, list):
        raise ValueError(f"Claim {claim_id} cited_doc_ids must be a list")
    return SciFactClaim(
        id=claim_id,
        text=text,
        evidence=_parse_evidence(claim_id, row["evidence"]),
        cited_doc_ids=tuple(_identifier(doc_id, label="Cited doc id") for doc_id in cited_raw),
    )


def _read_jsonl(path: Path) -> list[object]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    rows: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number} is not JSON") from exc
    return rows


def load_claims(path: Path, *, split: ClaimSplit) -> tuple[SciFactClaim, ...]:
    claims = tuple(parse_claim(row, split=split) for row in _read_jsonl(path))
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError(f"Duplicate claim ids in {path}")
    if not claims:
        raise ValueError(f"No claims in {path}")
    return claims


def load_abstracts(path: Path) -> dict[str, AbstractDocument]:
    documents: dict[str, AbstractDocument] = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError("A corpus row must be a JSON object")
        document_id = _identifier(row.get("doc_id"), label="Document id")
        title = row.get("title")
        abstract = row.get("abstract")
        if not isinstance(title, str) or not isinstance(abstract, list):
            raise ValueError(f"Document {document_id} is missing a title or abstract")
        sentences: list[str] = []
        for sentence in abstract:
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"Document {document_id} has an empty abstract sentence")
            sentences.append(sentence)
        if document_id in documents:
            raise ValueError(f"Duplicate document id {document_id}")
        documents[document_id] = AbstractDocument(
            id=document_id, title=title, sentences=tuple(sentences)
        )
    if not documents:
        raise ValueError(f"No abstracts in {path}")
    return documents


def check_sentence_indexes(
    claims: Sequence[SciFactClaim], abstracts: Mapping[str, AbstractDocument]
) -> None:
    """Sentence IDs index the abstract list. They are rejected when they fall outside it."""
    for claim in claims:
        if claim.labels_withheld:
            continue
        for item in claim.evidence:
            abstract = abstracts.get(item.document_id)
            if abstract is None:
                raise ValueError(f"Claim {claim.id} cites missing abstract {item.document_id}")
            if item.sentence >= len(abstract.sentences):
                raise ValueError(
                    f"Claim {claim.id} sentence {item.sentence} is outside abstract {item.document_id}"
                )


def align_split(
    claims: Sequence[SciFactClaim],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]] | None,
    *,
    claim_split: str,
    beir_qrel_split: str | None,
) -> SplitAlignment:
    """Compare claim IDs and texts with BEIR queries. Qrels are grades, not verdicts."""
    text_matches = 0
    missing = 0
    cited_equal = 0
    evidence_equal = 0
    evidence_subset = 0
    grades: set[int] = set()
    for claim in claims:
        query = queries.get(claim.id)
        if query is None:
            missing += 1
        elif query == claim.text:
            text_matches += 1
        if qrels is None:
            continue
        graded = qrels.get(claim.id, {})
        grades.update(graded.values())
        qrel_ids = set(graded)
        if qrel_ids == set(claim.cited_doc_ids):
            cited_equal += 1
        if qrel_ids == claim.evidence_document_ids():
            evidence_equal += 1
        if claim.evidence_document_ids() <= set(claim.cited_doc_ids):
            evidence_subset += 1
    return SplitAlignment(
        claim_split=claim_split,
        beir_qrel_split=beir_qrel_split,
        claim_count=len(claims),
        query_text_matches=text_matches,
        claims_missing_from_queries=missing,
        qrel_equals_cited=cited_equal,
        qrel_equals_evidence=evidence_equal,
        evidence_subset_of_cited=evidence_subset,
        grades=tuple(sorted(grades)),
    )


def _require_labeled_split(
    report: SplitAlignment, qrels: Mapping[str, Mapping[str, int]], claims: Sequence[SciFactClaim]
) -> None:
    ids = {claim.id for claim in claims}
    if any(claim.labels_withheld for claim in claims):
        raise ValueError(f"{report.claim_split} claims must carry labels")
    if report.claims_missing_from_queries or report.query_text_matches != report.claim_count:
        raise ValueError(f"{report.claim_split} claim IDs or texts do not match the BEIR queries")
    if set(qrels) != ids:
        raise ValueError(
            f"{report.claim_split} claim IDs do not match the {report.beir_qrel_split} qrel IDs"
        )
    if report.grades != (1,):
        raise ValueError(
            f"{report.beir_qrel_split} qrel grades are {report.grades}, expected only 1"
        )
    if report.qrel_equals_cited != report.claim_count:
        raise ValueError(f"{report.beir_qrel_split} qrels are not the cited document IDs")
    if report.evidence_subset_of_cited != report.claim_count:
        raise ValueError(
            f"{report.claim_split} evidence documents are not a subset of cited documents"
        )
    if report.qrel_equals_evidence == report.claim_count:
        raise ValueError(
            "BEIR qrels matched every evidence-document set. Refusing to treat those grades as verdicts"
        )


def require_scifact_beir_mapping(
    train: Sequence[SciFactClaim],
    dev: Sequence[SciFactClaim],
    official_test: Sequence[SciFactClaim],
    queries: Mapping[str, str],
    train_qrels: Mapping[str, Mapping[str, int]],
    beir_test_qrels: Mapping[str, Mapping[str, int]],
) -> MappingReport:
    """Check the claim-file to BEIR-query mapping. Dev claims line up with the BEIR test qrels.

    Train claim IDs equal the BEIR train query IDs and the claim text matches.
    Dev claim IDs equal the BEIR test query IDs. The public SciFact test IDs
    are not in the BEIR query file. Qrel document sets equal ``cited_doc_ids``.
    They are not the evidence-document sets, and the grade is the integer 1.
    """
    train_report = align_split(
        train, queries, train_qrels, claim_split="train", beir_qrel_split="train"
    )
    dev_report = align_split(
        dev, queries, beir_test_qrels, claim_split="dev", beir_qrel_split="test"
    )
    test_report = align_split(
        official_test, queries, None, claim_split="test", beir_qrel_split=None
    )
    _require_labeled_split(train_report, train_qrels, train)
    _require_labeled_split(dev_report, beir_test_qrels, dev)
    if not official_test or not all(claim.labels_withheld for claim in official_test):
        raise ValueError("Public test claims must be present and unlabeled")
    if test_report.claims_missing_from_queries != test_report.claim_count:
        raise ValueError("A public test claim ID appears in the BEIR queries")
    train_ids = {claim.id for claim in train}
    dev_ids = {claim.id for claim in dev}
    test_ids = {claim.id for claim in official_test}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError("SciFact claim splits overlap")
    if set(queries) != train_ids | dev_ids:
        raise ValueError("BEIR queries are not exactly the SciFact train and dev claims")
    return MappingReport(
        train=train_report, dev=dev_report, official_test=test_report, query_count=len(queries)
    )


def load_beir_corpus_fields(path: Path) -> dict[str, tuple[str, str]]:
    """Return BEIR ``_id`` to ``(title, text)``. The text is the body, not a sentence array."""
    documents: dict[str, tuple[str, str]] = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError(f"{path} has a corpus row that is not an object")
        document_id = str(row.get("_id", "")).strip()
        title = row.get("title", "")
        body = row.get("text", "")
        if not document_id or not isinstance(title, str) or not isinstance(body, str):
            raise ValueError(f"{path} has a corpus row without an id, title, and text")
        if document_id in documents:
            raise ValueError(f"Duplicate BEIR document id {document_id}")
        documents[document_id] = (title, body)
    if not documents:
        raise ValueError(f"No BEIR documents in {path}")
    return documents


def collapsed_whitespace(text: str) -> str:
    return " ".join(text.split())


def compare_abstracts_to_beir_bodies(
    abstracts: Mapping[str, AbstractDocument], beir_documents: Mapping[str, tuple[str, str]]
) -> tuple[int, int]:
    """Return exact space-join mismatches and whitespace-normalized mismatches.

    ``beir_documents`` maps a document ID to ``(title, body)``. Sentence IDs
    index ``abstracts``, not a split of the BEIR body. After whitespace
    collapsing, the body is the abstract sentences joined by spaces.
    """
    if set(abstracts) != set(beir_documents):
        raise ValueError("SciFact abstract IDs do not match the BEIR corpus IDs")
    exact = 0
    normalized = 0
    for document_id, abstract in abstracts.items():
        title, body = beir_documents[document_id]
        if title != abstract.title:
            raise ValueError(f"Title mismatch for document {document_id}")
        joined = " ".join(abstract.sentences)
        if body != joined:
            exact += 1
        if collapsed_whitespace(body) != collapsed_whitespace(joined):
            normalized += 1
    return exact, normalized


def extract_scifact_archive(archive: Path, destination: Path) -> None:
    """Extract the four claim and corpus files. Other members are ignored."""
    destination.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe tar member: {member.name}")
            target_name = _ARCHIVE_MEMBERS.get(member.name)
            if target_name is None:
                continue
            if not member.isfile():
                raise ValueError(f"Expected a file at {member.name}")
            extracted = package.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read {member.name}")
            (destination / target_name).write_bytes(extracted.read())
            found.add(member.name)
    missing = sorted(set(_ARCHIVE_MEMBERS) - found)
    if missing:
        raise ValueError(f"SciFact archive is missing {missing}")


class SciFactRelease(ConfigModel):
    train: tuple[SciFactClaim, ...]
    dev: tuple[SciFactClaim, ...]
    test: tuple[SciFactClaim, ...]
    abstracts: dict[str, AbstractDocument]
    archive_sha256: str
    claims_train_sha256: str
    claims_dev_sha256: str
    claims_test_sha256: str
    corpus_sha256: str


def load_scifact_release(directory: Path, *, archive_sha256: str) -> SciFactRelease:
    """Load claims and abstracts already extracted into ``directory``."""
    train = load_claims(directory / "claims_train.jsonl", split="train")
    dev = load_claims(directory / "claims_dev.jsonl", split="dev")
    test = load_claims(directory / "claims_test.jsonl", split="test")
    abstracts = load_abstracts(directory / "corpus.jsonl")
    check_sentence_indexes((*train, *dev), abstracts)
    return SciFactRelease(
        train=train,
        dev=dev,
        test=test,
        abstracts=abstracts,
        archive_sha256=archive_sha256,
        claims_train_sha256=file_sha256(directory / "claims_train.jsonl"),
        claims_dev_sha256=file_sha256(directory / "claims_dev.jsonl"),
        claims_test_sha256=file_sha256(directory / "claims_test.jsonl"),
        corpus_sha256=file_sha256(directory / "corpus.jsonl"),
    )


def prepare_scifact_release(
    cache: Path,
    *,
    sha256: str = SCIFACT_CLAIMS_SHA256,
    urls: tuple[str, ...] = (SCIFACT_CLAIMS_URL,),
) -> SciFactRelease:
    """Download the AllenAI tarball when needed, verify it, and parse the claim files."""
    root = cache.resolve()
    archive = root / "data.tar.gz"
    extracted = root / "data"
    _download(urls, archive, sha256)
    extract_scifact_archive(archive, extracted)
    return load_scifact_release(extracted, archive_sha256=sha256)
