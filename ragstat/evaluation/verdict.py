"""Train-selected SciFact verdicts and evidence sentences.

Retrieval is the frozen document-level BM25 index. It is not retuned here.
An NLI scorer labels abstract sentences from the top retrieved documents.
Knobs are chosen on the train claims. The dev claims are the held-out labeled
split because the public SciFact test file withholds evidence labels.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal

import yaml
from pydantic import Field, model_validator

from ragstat.config import (
    BM25Config,
    ChunkingConfig,
    ConfigModel,
    Nonblank,
    PositiveInt,
    SchemaVersion,
    UnitFloat,
)
from ragstat.datasets.scifact_claims import AbstractDocument, SciFactClaim
from ragstat.evaluation.comparison import QuestionChange
from ragstat.evaluation.factual import (
    VERDICTS,
    accuracy,
    class_f1,
    evidence_only_sentence_f1,
    macro_f1,
    micro_sentence_scores,
    sentence_scores,
)
from ragstat.evaluation.nli import NliModelConfig, NliScorer, NliScores, classify_nli
from ragstat.evaluation.statistics import bootstrap_mean_ci, signflip_p_value
from ragstat.evaluation.timing import percentile
from ragstat.ingestion.chunker import chunk_documents
from ragstat.ingestion.loader import Document
from ragstat.retrieval.bm25 import BM25Retriever

TIE_BREAK = (
    "higher mean sentence F1",
    "smaller doc_k",
    "smaller sentence_k",
    "higher min_confidence",
)
MAJORITY_TIE_ORDER = ("NEI", "SUPPORT", "CONTRADICT")


class VerdictPolicy(ConfigModel):
    doc_k: PositiveInt
    sentence_k: PositiveInt
    min_confidence: UnitFloat


class FrozenRetrieval(ConfigModel):
    type: Literal["bm25"] = "bm25"
    k1: float = Field(strict=True, gt=0, allow_inf_nan=False)
    b: UnitFloat
    tokenizer: Literal["stem"]
    unit: Literal["document"]


class VerdictGrid(ConfigModel):
    doc_k: tuple[PositiveInt, ...]
    sentence_k: tuple[PositiveInt, ...]
    min_confidence: tuple[UnitFloat, ...]

    @model_validator(mode="after")
    def values_are_distinct(self) -> "VerdictGrid":
        for name in ("doc_k", "sentence_k", "min_confidence"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be distinct")
        return self


class VerdictExperiment(ConfigModel):
    schema_version: SchemaVersion = 1
    claims_cache: Path
    documents_path: Path
    beir_raw: Path
    split: Literal["train", "dev"]
    retrieval: FrozenRetrieval
    nli: NliModelConfig
    grid: VerdictGrid
    policy: VerdictPolicy | None = None
    baseline_verdict: Literal["SUPPORT", "CONTRADICT", "NEI"] | None = None

    @model_validator(mode="after")
    def dev_carries_the_frozen_choice(self) -> "VerdictExperiment":
        frozen = self.policy is not None and self.baseline_verdict is not None
        if self.split == "dev" and not frozen:
            raise ValueError("The dev experiment must name a frozen policy and baseline verdict")
        if self.split == "train" and (self.policy is not None or self.baseline_verdict is not None):
            raise ValueError("The train experiment selects a policy and does not freeze one")
        return self


def load_verdict_experiment(path: Path) -> VerdictExperiment:
    """Load a verdict YAML file and resolve its paths relative to that file."""
    path = path.resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read verdict configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Verdict configuration {path} must be a YAML mapping")
    experiment = VerdictExperiment.model_validate(raw)
    return experiment.model_copy(
        update={
            "claims_cache": (path.parent / experiment.claims_cache).resolve(),
            "documents_path": (path.parent / experiment.documents_path).resolve(),
            "beir_raw": (path.parent / experiment.beir_raw).resolve(),
        }
    )


def assert_train_selection(split: str) -> None:
    if split != "train":
        raise ValueError("Verdict knobs are selected on the SciFact train claims only")


class ClaimPrediction(ConfigModel):
    verdict: Literal["SUPPORT", "CONTRADICT", "NEI"]
    sentences: tuple[tuple[Nonblank, int], ...] = ()

    @model_validator(mode="after")
    def sentences_are_unique(self) -> "ClaimPrediction":
        if len(set(self.sentences)) != len(self.sentences):
            raise ValueError("Predicted evidence sentences must be unique")
        if any(index < 0 for _document_id, index in self.sentences):
            raise ValueError("Predicted sentence indexes must be nonnegative")
        return self


class ClaimScore(ConfigModel):
    id: Nonblank
    gold: Literal["SUPPORT", "CONTRADICT", "NEI"]
    predicted: Literal["SUPPORT", "CONTRADICT", "NEI"]
    correct: float
    gold_sentences: tuple[str, ...]
    predicted_sentences: tuple[str, ...]
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float


class VerdictEvaluation(ConfigModel):
    schema_version: Literal[1] = 1
    split: Literal["train", "dev"]
    system: Literal["nli", "majority_baseline", "oracle_retrieval"]
    claim_count: int
    gold_counts: dict[str, int]
    accuracy: float
    macro_f1: float
    class_f1: dict[str, float]
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float
    micro_sentence_precision: float
    micro_sentence_recall: float
    micro_sentence_f1: float
    evidence_only_sentence_f1: float | None = None
    doc_k: int | None = None
    sentence_k: int | None = None
    min_confidence: float | None = None
    baseline_verdict: str | None = None
    model_name: str
    model_revision: str
    pair_order: str
    retrieval_p95_ms: float | None = None
    nli_pairs: int | None = None
    nli_truncated_pairs: int | None = None
    nli_seconds: float | None = None
    claims_sha256: str = ""
    corpus_sha256: str = ""
    environment: dict[str, str] = Field(default_factory=dict)
    claims: tuple[ClaimScore, ...]

    @model_validator(mode="after")
    def aggregates_match_claims(self) -> "VerdictEvaluation":
        if self.claim_count != len(self.claims):
            raise ValueError("claim_count does not match the claim rows")
        if set(self.gold_counts) != set(VERDICTS) or set(self.class_f1) != set(VERDICTS):
            raise ValueError("gold_counts and class_f1 must name SUPPORT, CONTRADICT, and NEI")
        gold: list[str] = [claim.gold for claim in self.claims]
        predicted: list[str] = [claim.predicted for claim in self.claims]
        if any(self.gold_counts[label] != gold.count(label) for label in VERDICTS):
            raise ValueError("gold_counts do not match the claim rows")
        for claim in self.claims:
            precision, recall, f1 = sentence_scores(
                _parse_sentences(claim.gold_sentences),
                _parse_sentences(claim.predicted_sentences),
            )
            if (
                claim.correct != float(claim.gold == claim.predicted)
                or not _close(claim.sentence_precision, precision)
                or not _close(claim.sentence_recall, recall)
                or not _close(claim.sentence_f1, f1)
            ):
                raise ValueError(f"Claim {claim.id} scores do not match its sentence sets")
        if not _close(self.accuracy, accuracy(gold, predicted)):
            raise ValueError("accuracy does not match the claim rows")
        if not _close(self.macro_f1, macro_f1(gold, predicted)):
            raise ValueError("macro_f1 does not match the claim rows")
        for label in VERDICTS:
            if not _close(self.class_f1[label], class_f1(gold, predicted, label)):
                raise ValueError(f"class_f1 for {label} does not match the claim rows")
        if not _close(
            self.sentence_precision, fmean(claim.sentence_precision for claim in self.claims)
        ):
            raise ValueError("sentence_precision does not match the claim rows")
        if not _close(self.sentence_recall, fmean(claim.sentence_recall for claim in self.claims)):
            raise ValueError("sentence_recall does not match the claim rows")
        if not _close(self.sentence_f1, fmean(claim.sentence_f1 for claim in self.claims)):
            raise ValueError("sentence_f1 does not match the claim rows")
        micro = micro_sentence_scores(
            [_parse_sentences(claim.gold_sentences) for claim in self.claims],
            [_parse_sentences(claim.predicted_sentences) for claim in self.claims],
        )
        if not all(
            _close(reported, expected)
            for reported, expected in zip(
                (
                    self.micro_sentence_precision,
                    self.micro_sentence_recall,
                    self.micro_sentence_f1,
                ),
                micro,
                strict=True,
            )
        ):
            raise ValueError("micro sentence scores do not match the claim rows")
        evidence_only = evidence_only_sentence_f1(
            [_parse_sentences(claim.gold_sentences) for claim in self.claims],
            [_parse_sentences(claim.predicted_sentences) for claim in self.claims],
        )
        if self.evidence_only_sentence_f1 is not None and (
            evidence_only is None or not _close(self.evidence_only_sentence_f1, evidence_only)
        ):
            raise ValueError("evidence_only_sentence_f1 does not match the claim rows")
        return self


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def _parse_sentences(values: Sequence[str]) -> set[tuple[str, int]]:
    parsed: set[tuple[str, int]] = set()
    for value in values:
        document_id, raw_index = value.rsplit(":", 1)
        if not document_id or not raw_index.isdigit() or (document_id, int(raw_index)) in parsed:
            raise ValueError(f"Bad evidence sentence id {value}")
        parsed.add((document_id, int(raw_index)))
    return parsed


def format_sentence(document_id: str, index: int) -> str:
    return f"{document_id}:{index}"


def _claim_order(claim_id: str) -> tuple[int, int | str]:
    return (0, int(claim_id)) if claim_id.isdigit() else (1, claim_id)


def _sentence_order(item: tuple[str, int]) -> tuple[int, int | str, int]:
    kind, document = _claim_order(item[0])
    return (kind, document, item[1])


def majority_verdict(labels: Sequence[str]) -> tuple[str, dict[str, int]]:
    """Most common train label. Ties prefer NEI, then SUPPORT, then CONTRADICT."""
    counts = dict.fromkeys(VERDICTS, 0)
    for label in labels:
        if label not in counts:
            raise ValueError(f"Unknown verdict {label}")
        counts[label] += 1
    if not labels:
        raise ValueError("Majority verdict requires at least one label")
    best = max(counts.values())
    chosen = min((label for label in counts if counts[label] == best), key=MAJORITY_TIE_ORDER.index)
    return chosen, counts


def evaluate_predictions(
    claims: Sequence[SciFactClaim],
    predictions: Mapping[str, ClaimPrediction],
    *,
    split: Literal["train", "dev"],
    system: Literal["nli", "majority_baseline", "oracle_retrieval"],
    model_name: str,
    model_revision: str,
    pair_order: str,
    doc_k: int | None = None,
    sentence_k: int | None = None,
    min_confidence: float | None = None,
    baseline_verdict: str | None = None,
) -> VerdictEvaluation:
    """Score labeled claims. Public test claims are rejected because their labels are withheld."""
    if {claim.id for claim in claims} != set(predictions):
        raise ValueError("Predictions must cover each claim once")
    rows: list[ClaimScore] = []
    for claim in sorted(claims, key=lambda item: _claim_order(item.id)):
        prediction = predictions[claim.id]
        gold_sentences = claim.gold_sentences()
        predicted_sentences = set(prediction.sentences)
        precision, recall, f1 = sentence_scores(set(gold_sentences), predicted_sentences)
        rows.append(
            ClaimScore(
                id=claim.id,
                gold=claim.gold_verdict(),
                predicted=prediction.verdict,
                correct=float(claim.gold_verdict() == prediction.verdict),
                gold_sentences=tuple(
                    format_sentence(document_id, index)
                    for document_id, index in sorted(gold_sentences, key=_sentence_order)
                ),
                predicted_sentences=tuple(
                    format_sentence(document_id, index)
                    for document_id, index in sorted(predicted_sentences, key=_sentence_order)
                ),
                sentence_precision=precision,
                sentence_recall=recall,
                sentence_f1=f1,
            )
        )
    gold: list[str] = [row.gold for row in rows]
    predicted: list[str] = [row.predicted for row in rows]
    gold_sets = [_parse_sentences(row.gold_sentences) for row in rows]
    predicted_sets = [_parse_sentences(row.predicted_sentences) for row in rows]
    micro = micro_sentence_scores(gold_sets, predicted_sets)
    return VerdictEvaluation(
        split=split,
        system=system,
        claim_count=len(rows),
        gold_counts={label: gold.count(label) for label in VERDICTS},
        accuracy=accuracy(gold, predicted),
        macro_f1=macro_f1(gold, predicted),
        class_f1={label: class_f1(gold, predicted, label) for label in VERDICTS},
        sentence_precision=fmean(row.sentence_precision for row in rows),
        sentence_recall=fmean(row.sentence_recall for row in rows),
        sentence_f1=fmean(row.sentence_f1 for row in rows),
        micro_sentence_precision=micro[0],
        micro_sentence_recall=micro[1],
        micro_sentence_f1=micro[2],
        evidence_only_sentence_f1=evidence_only_sentence_f1(gold_sets, predicted_sets),
        doc_k=doc_k,
        sentence_k=sentence_k,
        min_confidence=min_confidence,
        baseline_verdict=baseline_verdict,
        model_name=model_name,
        model_revision=model_revision,
        pair_order=pair_order,
        claims=tuple(rows),
    )


@dataclass(frozen=True)
class RetrievedSentence:
    document_id: str
    sentence: int
    retrieval_rank: int
    scores: NliScores


@dataclass(frozen=True)
class SentenceJob:
    claim_id: str
    document_id: str
    sentence: int
    retrieval_rank: int
    premise: str
    hypothesis: str


def premise_text(sentence: str) -> str:
    """One abstract sentence. Newlines become spaces. The index still points at the original list."""
    text = sentence.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    if not text:
        raise ValueError("Abstract sentence is empty")
    return text


def _verdict(label: str) -> Literal["SUPPORT", "CONTRADICT", "NEI"]:
    if label == "SUPPORT":
        return "SUPPORT"
    if label == "CONTRADICT":
        return "CONTRADICT"
    if label == "NEI":
        return "NEI"
    raise ValueError(f"Unknown verdict {label}")


def build_sentence_jobs(
    claims: Sequence[SciFactClaim],
    abstracts: Mapping[str, AbstractDocument],
    retrieved: Mapping[str, Sequence[str]],
    doc_k: int,
) -> list[SentenceJob]:
    """Pair each claim with sentences from its top ``doc_k`` retrieved abstracts."""
    jobs: list[SentenceJob] = []
    for claim in claims:
        for rank, document_id in enumerate(retrieved.get(claim.id, ())[:doc_k]):
            abstract = abstracts.get(document_id)
            if abstract is None:
                raise ValueError(f"Retrieved document {document_id} has no SciFact abstract")
            for index, sentence in enumerate(abstract.sentences):
                jobs.append(
                    SentenceJob(
                        claim_id=claim.id,
                        document_id=document_id,
                        sentence=index,
                        retrieval_rank=rank,
                        premise=premise_text(sentence),
                        hypothesis=claim.text,
                    )
                )
    return jobs


def oracle_document_lists(claims: Sequence[SciFactClaim]) -> dict[str, tuple[str, ...]]:
    """Gold evidence documents, sorted by id. NEI claims map to an empty tuple."""
    return {
        claim.id: tuple(sorted(claim.evidence_document_ids(), key=_claim_order)) for claim in claims
    }


def build_oracle_jobs(
    claims: Sequence[SciFactClaim],
    abstracts: Mapping[str, AbstractDocument],
) -> list[SentenceJob]:
    """Every gold evidence sentence, with retrieval rank 0.

    The frozen policy still applies ``sentence_k`` and ``min_confidence``. Rank 0
    keeps every gold document inside ``doc_k``. A claim with no gold evidence
    contributes no jobs, so the policy predicts NEI.
    """
    retrieved = oracle_document_lists(claims)
    widest = max((len(documents) for documents in retrieved.values()), default=1)
    jobs = build_sentence_jobs(claims, abstracts, retrieved, max(widest, 1))
    return [
        SentenceJob(
            claim_id=job.claim_id,
            document_id=job.document_id,
            sentence=job.sentence,
            retrieval_rank=0,
            premise=job.premise,
            hypothesis=job.hypothesis,
        )
        for job in jobs
    ]


def score_sentence_jobs(
    jobs: Sequence[SentenceJob], scorer: NliScorer
) -> dict[str, tuple[RetrievedSentence, ...]]:
    if not jobs:
        return {}
    scores = scorer.score([(job.premise, job.hypothesis) for job in jobs])
    if len(scores) != len(jobs):
        raise ValueError("Expected one NLI score per abstract sentence")
    grouped: dict[str, list[RetrievedSentence]] = {}
    for job, distribution in zip(jobs, scores, strict=True):
        grouped.setdefault(job.claim_id, []).append(
            RetrievedSentence(
                document_id=job.document_id,
                sentence=job.sentence,
                retrieval_rank=job.retrieval_rank,
                scores=distribution,
            )
        )
    return {claim_id: tuple(rows) for claim_id, rows in grouped.items()}


def apply_policy(rows: Sequence[RetrievedSentence], policy: VerdictPolicy) -> ClaimPrediction:
    """Keep non-neutral sentences at or above the confidence gate, then cut to ``sentence_k``.

    Kept sentences are ordered by descending probability, then document ID, then
    sentence index. The verdict is the label of the first kept sentence. When
    the kept sentences disagree, that is the highest-ranked label. No kept
    sentence means NEI and an empty evidence set.
    """
    kept: list[tuple[float, str, int, str]] = []
    for row in rows:
        if row.retrieval_rank >= policy.doc_k:
            continue
        label, confidence = classify_nli(row.scores)
        if label == "NEI" or confidence < policy.min_confidence:
            continue
        kept.append((confidence, row.document_id, row.sentence, label))
    kept.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = kept[: policy.sentence_k]
    if not chosen:
        return ClaimPrediction(verdict="NEI", sentences=())
    return ClaimPrediction(
        verdict=_verdict(chosen[0][3]),
        sentences=tuple((document_id, index) for _confidence, document_id, index, _label in chosen),
    )


def predict_baseline(
    claims: Sequence[SciFactClaim],
    retrieved: Mapping[str, Sequence[str]],
    abstracts: Mapping[str, AbstractDocument],
    verdict: str,
) -> dict[str, ClaimPrediction]:
    """Majority verdict. Evidence is sentence 0 of the top document, except when the verdict is NEI."""
    label = _verdict(verdict)
    predictions: dict[str, ClaimPrediction] = {}
    for claim in claims:
        documents = retrieved.get(claim.id, ())
        sentences: tuple[tuple[str, int], ...] = ()
        if label != "NEI" and documents:
            abstract = abstracts.get(documents[0])
            if abstract is None:
                raise ValueError(f"Retrieved document {documents[0]} has no SciFact abstract")
            sentences = ((documents[0], 0),)
        predictions[claim.id] = ClaimPrediction(verdict=label, sentences=sentences)
    return predictions


def predictions_for_policy(
    claims: Sequence[SciFactClaim],
    rows: Mapping[str, Sequence[RetrievedSentence]],
    policy: VerdictPolicy,
) -> dict[str, ClaimPrediction]:
    return {claim.id: apply_policy(rows.get(claim.id, ()), policy) for claim in claims}


def iter_policies(grid: VerdictGrid) -> tuple[VerdictPolicy, ...]:
    return tuple(
        VerdictPolicy(doc_k=doc_k, sentence_k=sentence_k, min_confidence=threshold)
        for doc_k in grid.doc_k
        for sentence_k in grid.sentence_k
        for threshold in grid.min_confidence
    )


def choose_policy(
    scored: Sequence[tuple[VerdictPolicy, VerdictEvaluation]],
) -> tuple[VerdictPolicy, VerdictEvaluation]:
    """Max train macro-F1. Ties follow ``TIE_BREAK``."""
    if not scored:
        raise ValueError("Policy selection requires at least one candidate")
    return max(
        scored,
        key=lambda item: (
            item[1].macro_f1,
            item[1].sentence_f1,
            -item[0].doc_k,
            -item[0].sentence_k,
            item[0].min_confidence,
        ),
    )


def retrieve_top_documents(
    documents: Sequence[Document],
    queries: Sequence[tuple[str, str]],
    *,
    k: int,
    k1: float,
    b: float,
) -> tuple[dict[str, tuple[str, ...]], list[float]]:
    """Document-level stem BM25. Returns document IDs and per-query milliseconds.

    The clock covers the full-corpus score, which is what document-level BM25
    does before it slices to ``k``.
    """
    chunks = chunk_documents(documents, ChunkingConfig(unit="document"))
    retriever = BM25Retriever(BM25Config(k1=k1, b=b, tokenizer="stem"))
    retriever.index(chunks)
    found: dict[str, tuple[str, ...]] = {}
    elapsed: list[float] = []
    for claim_id, text in queries:
        started = perf_counter()
        hits = retriever.retrieve(text, k)
        elapsed.append((perf_counter() - started) * 1000)
        found[claim_id] = tuple(hit.chunk.document_id for hit in hits)
    return found, elapsed


class VerdictErrorBreakdown(ConfigModel):
    """Where frozen-policy mistakes come from. This is a dev measurement, not a selection."""

    claim_count: int
    correct: int
    incorrect: int
    support_to_nei: int
    support_to_nei_evidence_at_rank_1: int
    support_to_nei_evidence_missed: int
    incorrect_with_gold_hit: int
    incorrect_with_gold_miss: int
    incorrect_without_gold_evidence: int

    @model_validator(mode="after")
    def counts_partition_the_claims(self) -> "VerdictErrorBreakdown":
        if self.correct + self.incorrect != self.claim_count:
            raise ValueError("correct and incorrect must add up to claim_count")
        if (
            self.incorrect_with_gold_hit
            + self.incorrect_with_gold_miss
            + self.incorrect_without_gold_evidence
            != self.incorrect
        ):
            raise ValueError("incorrect rows must partition into hit, miss, and no gold evidence")
        if (
            self.support_to_nei_evidence_at_rank_1 + self.support_to_nei_evidence_missed
            != self.support_to_nei
        ):
            raise ValueError("SUPPORT to NEI rows must partition into rank-1 hits and misses")
        return self


def _gold_document_ids(sentences: Sequence[str]) -> set[str]:
    return {value.rsplit(":", 1)[0] for value in sentences}


def breakdown_errors(
    claims: Sequence[ClaimScore],
    rankings: Mapping[str, Sequence[str]],
    *,
    doc_k: int,
) -> VerdictErrorBreakdown:
    """Split mistakes by whether a gold evidence document is inside the top ``doc_k``.

    Rank 1 is the first retrieved document. Claims with no gold evidence, which
    are the NEI labels, are not retrieval misses.
    """
    if isinstance(doc_k, bool) or not isinstance(doc_k, int) or doc_k <= 0:
        raise ValueError("doc_k must be a positive integer")
    if {claim.id for claim in claims} != set(rankings):
        raise ValueError("Rankings must cover each claim once")
    correct = 0
    support_hit = support_miss = 0
    incorrect_hit = incorrect_miss = incorrect_empty = 0
    for claim in claims:
        gold_documents = _gold_document_ids(claim.gold_sentences)
        hit = any(document_id in gold_documents for document_id in rankings[claim.id][:doc_k])
        if claim.gold == claim.predicted:
            correct += 1
            continue
        if not gold_documents:
            incorrect_empty += 1
        elif hit:
            incorrect_hit += 1
        else:
            incorrect_miss += 1
        if claim.gold == "SUPPORT" and claim.predicted == "NEI":
            if hit:
                support_hit += 1
            else:
                support_miss += 1
    incorrect = len(claims) - correct
    return VerdictErrorBreakdown(
        claim_count=len(claims),
        correct=correct,
        incorrect=incorrect,
        support_to_nei=support_hit + support_miss,
        support_to_nei_evidence_at_rank_1=support_hit,
        support_to_nei_evidence_missed=support_miss,
        incorrect_with_gold_hit=incorrect_hit,
        incorrect_with_gold_miss=incorrect_miss,
        incorrect_without_gold_evidence=incorrect_empty,
    )


class MetricSnapshot(ConfigModel):
    accuracy: float
    macro_f1: float
    micro_sentence_f1: float
    evidence_only_sentence_f1: float | None
    sentence_f1: float


def metric_snapshot(report: VerdictEvaluation) -> MetricSnapshot:
    return MetricSnapshot(
        accuracy=report.accuracy,
        macro_f1=report.macro_f1,
        micro_sentence_f1=report.micro_sentence_f1,
        evidence_only_sentence_f1=report.evidence_only_sentence_f1,
        sentence_f1=report.sentence_f1,
    )


class VerdictDevAnalysis(ConfigModel):
    """Held-out dev measurement. No knob in this file was chosen on dev."""

    split: Literal["dev"] = "dev"
    selected_on_this_split: Literal[False] = False
    doc_k: PositiveInt
    frozen_nli: MetricSnapshot
    majority_baseline: MetricSnapshot
    oracle_retrieval: MetricSnapshot
    errors: VerdictErrorBreakdown
    note: Nonblank


class VerdictMetricCheck(ConfigModel):
    metric: str
    baseline: float
    candidate: float
    mean_delta: float
    ci_low: float
    ci_high: float
    permutation_p_value: float | None
    rule: str


class VerdictComparison(ConfigModel):
    """Paired uncertainty. This file is a measurement. It is not a quality gate."""

    split: Literal["train", "dev"]
    seed: int = Field(ge=0)
    bootstrap_samples: PositiveInt
    permutation_samples: PositiveInt
    confidence: float = Field(gt=0, lt=1)
    checks: tuple[VerdictMetricCheck, ...]
    claim_changes: tuple[QuestionChange, ...]


def _direction(delta: float) -> Literal["improved", "regressed", "unchanged"]:
    if abs(delta) <= 1e-12:
        return "unchanged"
    return "improved" if delta > 0 else "regressed"


def _per_claim_value(score: ClaimScore, metric: str) -> float:
    if metric == "accuracy":
        return score.correct
    if metric == "sentence_precision":
        return score.sentence_precision
    if metric == "sentence_recall":
        return score.sentence_recall
    if metric == "sentence_f1":
        return score.sentence_f1
    raise ValueError(f"Unknown per-claim metric {metric}")


def bootstrap_macro_f1_delta(
    gold: Sequence[str],
    baseline: Sequence[str],
    candidate: Sequence[str],
    *,
    samples: int,
    seed: str,
    confidence: float,
) -> tuple[float, float, float]:
    """Resample claims and recompute macro-F1. This is not a mean of per-claim F1."""
    point = macro_f1(gold, candidate) - macro_f1(gold, baseline)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("Bootstrap sample count must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one")
    rng = random.Random(seed)
    size = len(gold)
    deltas: list[float] = []
    for _ in range(samples):
        index = [rng.randrange(size) for _ in range(size)]
        sampled_gold = [gold[item] for item in index]
        sampled_baseline = [baseline[item] for item in index]
        sampled_candidate = [candidate[item] for item in index]
        deltas.append(
            macro_f1(sampled_gold, sampled_candidate) - macro_f1(sampled_gold, sampled_baseline)
        )
    alpha = (1.0 - confidence) / 2.0
    return point, percentile(deltas, alpha), percentile(deltas, 1.0 - alpha)


def compare_verdicts(
    baseline: VerdictEvaluation,
    candidate: VerdictEvaluation,
    *,
    seed: int = 0,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 10_000,
    confidence: float = 0.95,
) -> VerdictComparison:
    """Candidate minus baseline on the same claims.

    Accuracy and the sentence means use the shared per-question bootstrap and
    sign-flip. Macro-F1 is recomputed on each resampled claim list, so it has
    a percentile interval and no sign-flip p-value.
    """
    if baseline.split != candidate.split:
        raise ValueError("Verdict reports must use the same split")
    if [claim.id for claim in baseline.claims] != [claim.id for claim in candidate.claims]:
        raise ValueError("Verdict reports must list the same claims in the same order")
    checks: list[VerdictMetricCheck] = []
    changes: list[QuestionChange] = []
    for metric in ("accuracy", "sentence_precision", "sentence_recall", "sentence_f1"):
        deltas = [
            _per_claim_value(new, metric) - _per_claim_value(old, metric)
            for old, new in zip(baseline.claims, candidate.claims, strict=True)
        ]
        mean_delta, low, high = bootstrap_mean_ci(
            deltas,
            samples=bootstrap_samples,
            seed=f"ragbench-stats-v1:{seed}:bootstrap:{metric}",
            confidence=confidence,
        )
        p_value = signflip_p_value(
            deltas,
            samples=permutation_samples,
            seed=f"ragbench-stats-v1:{seed}:permutation:{metric}",
        )
        base_value = fmean(_per_claim_value(claim, metric) for claim in baseline.claims)
        cand_value = fmean(_per_claim_value(claim, metric) for claim in candidate.claims)
        checks.append(
            VerdictMetricCheck(
                metric=metric,
                baseline=base_value,
                candidate=cand_value,
                mean_delta=mean_delta,
                ci_low=low,
                ci_high=high,
                permutation_p_value=p_value,
                rule="mean of per-claim scores; percentile bootstrap and sign-flip",
            )
        )
        for old, new, delta in zip(baseline.claims, candidate.claims, deltas, strict=True):
            changes.append(
                QuestionChange(
                    question_id=old.id,
                    metric=metric,
                    baseline=_per_claim_value(old, metric),
                    candidate=_per_claim_value(new, metric),
                    delta=delta,
                    direction=_direction(delta),
                )
            )
    gold = [claim.gold for claim in baseline.claims]
    mean_delta, low, high = bootstrap_macro_f1_delta(
        gold,
        [claim.predicted for claim in baseline.claims],
        [claim.predicted for claim in candidate.claims],
        samples=bootstrap_samples,
        seed=f"ragbench-stats-v1:{seed}:bootstrap:macro_f1",
        confidence=confidence,
    )
    checks.append(
        VerdictMetricCheck(
            metric="macro_f1",
            baseline=baseline.macro_f1,
            candidate=candidate.macro_f1,
            mean_delta=mean_delta,
            ci_low=low,
            ci_high=high,
            permutation_p_value=None,
            rule="macro-F1 recomputed on each resampled claim list; no sign-flip p-value",
        )
    )
    return VerdictComparison(
        split=baseline.split,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        confidence=confidence,
        checks=tuple(checks),
        claim_changes=tuple(sorted(changes, key=lambda item: (item.metric, item.question_id))),
    )
