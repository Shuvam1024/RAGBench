"""SciFact claim files joined to BEIR queries by claim ID. Qrels are not verdicts."""

import json
import tarfile
from pathlib import Path

import pytest

from ragbench.datasets.beir import _read_qrels, file_sha256
from ragbench.datasets.scifact_claims import (
    AbstractDocument,
    align_split,
    collapsed_whitespace,
    compare_abstracts_to_beir_bodies,
    extract_scifact_archive,
    load_beir_corpus_fields,
    load_claims,
    load_scifact_release,
    parse_claim,
    prepare_scifact_release,
    require_scifact_beir_mapping,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "scifact_beir_mapping.json").read_text(encoding="utf-8")
)
ROOT = Path(__file__).resolve().parents[1]


def _claims(split: str) -> tuple:
    return tuple(parse_claim(row, split=split) for row in FIXTURE[split])


def test_excerpt_maps_claim_ids_and_refuses_qrels_as_verdicts() -> None:
    train, dev, official = _claims("train"), _claims("dev"), _claims("test")
    report = require_scifact_beir_mapping(
        train,
        dev,
        official,
        FIXTURE["queries"],
        FIXTURE["train_qrels"],
        FIXTURE["beir_test_qrels"],
    )
    assert report.train.beir_qrel_split == "train"
    assert report.dev.beir_qrel_split == "test"
    assert report.train.qrel_equals_cited == 3
    assert report.train.qrel_equals_evidence == 0
    assert report.dev.qrel_equals_cited == 2
    assert report.dev.qrel_equals_evidence == 1
    assert report.official_test.claims_missing_from_queries == 1
    assert report.train.grades == (1,)

    claim = next(item for item in train if item.id == "263")
    assert claim.gold_verdict() == "CONTRADICT"
    assert claim.gold_sentences() == {
        ("11328820", 7),
        ("11328820", 9),
        ("30041340", 0),
        ("30041340", 1),
        ("30041340", 11),
    }
    assert "14853989" in claim.cited_doc_ids
    assert "14853989" not in claim.evidence_document_ids()
    assert "14853989" in FIXTURE["train_qrels"]["263"]
    nei = next(item for item in train if item.id == "0")
    assert nei.gold_verdict() == "NEI"
    assert nei.gold_sentences() == frozenset()
    assert set(FIXTURE["train_qrels"]["0"]) == set(nei.cited_doc_ids)
    with pytest.raises(ValueError, match="withheld"):
        official[0].gold_verdict()


def test_mapping_rejects_a_verdict_shaped_qrel_file(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\tverdict\n0\t31715818\t1\tSUPPORT\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="three tab-separated"):
        _read_qrels(path)


def test_mapping_rejects_text_grade_and_evidence_collisions() -> None:
    train, dev, official = _claims("train"), _claims("dev"), _claims("test")
    queries = dict(FIXTURE["queries"])
    queries["0"] = "different text"
    with pytest.raises(ValueError, match="texts"):
        require_scifact_beir_mapping(
            train, dev, official, queries, FIXTURE["train_qrels"], FIXTURE["beir_test_qrels"]
        )
    evidence_qrels = {claim.id: dict.fromkeys(claim.evidence_document_ids(), 1) for claim in train}
    with pytest.raises(ValueError, match="cited|grades|verdicts"):
        require_scifact_beir_mapping(
            train, dev, official, FIXTURE["queries"], evidence_qrels, FIXTURE["beir_test_qrels"]
        )
    graded = {claim_id: dict.fromkeys(docs, 2) for claim_id, docs in FIXTURE["train_qrels"].items()}
    with pytest.raises(ValueError, match="grades"):
        require_scifact_beir_mapping(
            train, dev, official, FIXTURE["queries"], graded, FIXTURE["beir_test_qrels"]
        )


def test_parser_rejects_mixed_labels_and_labeled_test_rows() -> None:
    mixed = {
        "id": 9,
        "claim": "A claim.",
        "evidence": {
            "1": [{"sentences": [0], "label": "SUPPORT"}],
            "2": [{"sentences": [0], "label": "CONTRADICT"}],
        },
        "cited_doc_ids": [1, 2],
    }
    with pytest.raises(ValueError, match="both SUPPORT and CONTRADICT"):
        parse_claim(mixed, split="train")
    with pytest.raises(ValueError, match="not expected to carry"):
        parse_claim({"id": 7, "claim": "A claim.", "evidence": {}}, split="test")
    with pytest.raises(ValueError, match="bad sentence"):
        parse_claim(
            {
                "id": 9,
                "claim": "A claim.",
                "evidence": {"1": [{"sentences": [-1], "label": "SUPPORT"}]},
                "cited_doc_ids": [1],
            },
            split="dev",
        )


def test_abstract_sentence_ids_are_not_title_indexes() -> None:
    abstract = AbstractDocument(id="10", title="Title sentence.", sentences=("First.", "Second."))
    assert abstract.sentences[0] == "First."
    assert abstract.title not in abstract.sentences
    exact, normalized = compare_abstracts_to_beir_bodies(
        {"10": AbstractDocument(id="10", title="Title", sentences=("Hello\n world",))},
        {"10": ("Title", "Hello world")},
    )
    assert exact == 1
    assert normalized == 0
    assert collapsed_whitespace("Hello\n world") == "Hello world"
    with pytest.raises(ValueError, match="Title mismatch"):
        compare_abstracts_to_beir_bodies(
            {"10": abstract},
            {"10": ("Other", "First. Second.")},
        )
    with pytest.raises(ValueError, match="do not match"):
        compare_abstracts_to_beir_bodies(
            {"10": abstract}, {"11": ("Title sentence.", "First. Second.")}
        )


def test_archive_round_trip_and_unsafe_member(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "claims_train.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "claim": "A labeled claim.",
                "evidence": {"5": [{"sentences": [0], "label": "SUPPORT"}]},
                "cited_doc_ids": [5, 6],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "claims_dev.jsonl").write_text(
        json.dumps({"id": 2, "claim": "Another claim.", "evidence": {}, "cited_doc_ids": [5]})
        + "\n",
        encoding="utf-8",
    )
    (source / "claims_test.jsonl").write_text(
        json.dumps({"id": 3, "claim": "An unlabeled claim."}) + "\n",
        encoding="utf-8",
    )
    (source / "corpus.jsonl").write_text(
        json.dumps(
            {
                "doc_id": 5,
                "title": "A title",
                "abstract": ["The rationale sentence."],
                "structured": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for name in (
            "claims_train.jsonl",
            "claims_dev.jsonl",
            "claims_test.jsonl",
            "corpus.jsonl",
        ):
            package.add(source / name, arcname=f"data/{name}")
        note = tmp_path / "note.txt"
        note.write_text("ignored")
        package.add(note, arcname="data/cross_validation/fold_1/note.txt")
    digest = file_sha256(archive)
    release = prepare_scifact_release(cache, sha256=digest, urls=("file:///unused",))
    assert [claim.id for claim in release.train] == ["1"]
    assert release.train[0].gold_verdict() == "SUPPORT"
    assert release.dev[0].gold_verdict() == "NEI"
    assert release.test[0].labels_withheld
    assert release.abstracts["5"].sentences == ("The rationale sentence.",)
    assert release.archive_sha256 == digest

    slipped = tmp_path / "slipped.tar.gz"
    with tarfile.open(slipped, "w:gz") as package:
        package.add(source / "corpus.jsonl", arcname="../corpus.jsonl")
    with pytest.raises(ValueError, match="Unsafe tar member"):
        extract_scifact_archive(slipped, tmp_path / "out")


def test_sentence_index_outside_the_abstract_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "claims_train.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "claim": "A labeled claim.",
                "evidence": {"5": [{"sentences": [3], "label": "SUPPORT"}]},
                "cited_doc_ids": [5],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "claims_dev.jsonl").write_text(
        json.dumps({"id": 2, "claim": "Dev claim.", "evidence": {}, "cited_doc_ids": [5]}) + "\n",
        encoding="utf-8",
    )
    (directory / "claims_test.jsonl").write_text(
        json.dumps({"id": 3, "claim": "Test claim."}) + "\n",
        encoding="utf-8",
    )
    (directory / "corpus.jsonl").write_text(
        json.dumps({"doc_id": 5, "title": "T", "abstract": ["Only one."], "structured": False})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside abstract"):
        load_scifact_release(directory, archive_sha256="0" * 64)
    claims = load_claims(directory / "claims_dev.jsonl", split="dev")
    assert claims[0].gold_sentences() == frozenset()


def test_beir_corpus_reader_rejects_a_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    row = json.dumps({"_id": "5", "title": "T", "text": "Body"})
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_beir_corpus_fields(path)


@pytest.mark.skipif(
    not (ROOT / ".cache/scifact/data.tar.gz").is_file()
    or not (ROOT / ".cache/beir/scifact/raw/scifact/queries.jsonl").is_file(),
    reason="Official SciFact claims and the BEIR SciFact zip are not in the local cache",
)
def test_official_release_matches_beir_query_ids() -> None:
    from ragbench.datasets.beir import _read_queries

    release = prepare_scifact_release(ROOT / ".cache/scifact")
    raw = ROOT / ".cache/beir/scifact/raw/scifact"
    report = require_scifact_beir_mapping(
        release.train,
        release.dev,
        release.test,
        _read_queries(raw / "queries.jsonl"),
        _read_qrels(raw / "qrels/train.tsv"),
        _read_qrels(raw / "qrels/test.tsv"),
    )
    exact, normalized = compare_abstracts_to_beir_bodies(
        release.abstracts, load_beir_corpus_fields(raw / "corpus.jsonl")
    )
    assert report.train.claim_count == 809
    assert report.train.qrel_equals_cited == 809
    assert report.train.qrel_equals_evidence == 480
    assert report.dev.claim_count == 300
    assert report.dev.beir_qrel_split == "test"
    assert report.dev.qrel_equals_cited == 300
    assert report.dev.qrel_equals_evidence == 175
    assert report.official_test.claim_count == 300
    assert report.official_test.claims_missing_from_queries == 300
    assert report.query_count == 1109
    assert exact == 1055
    assert normalized == 0
    counts: dict[str, int] = {"SUPPORT": 0, "CONTRADICT": 0, "NEI": 0}
    for claim in release.train:
        counts[claim.gold_verdict()] += 1
    assert counts == {"SUPPORT": 332, "CONTRADICT": 173, "NEI": 304}


def test_align_split_counts_a_missing_query() -> None:
    train = _claims("train")
    report = align_split(train, {}, None, claim_split="train", beir_qrel_split=None)
    assert report.claims_missing_from_queries == 3
    assert report.query_text_matches == 0
