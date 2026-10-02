"""Versioned benchmark inputs and inspectable evaluation outputs."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isclose
from statistics import fmean
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragbench.config import (
    Nonblank,
    NonnegativeFloat,
    PositiveInt,
    RunConfig,
    SchemaVersion,
    UnitFloat,
)
from ragbench.generation.models import GeneratedAnswer, JudgeResult

PositiveGrade = PositiveInt


def _aligned_means[KeyT](
    aggregate: Mapping[KeyT, float] | None,
    rows: Sequence[Mapping[KeyT, float] | None],
    label: str,
) -> None:
    if aggregate is None:
        if any(row is not None for row in rows):
            raise ValueError(f"Missing aggregate {label}")
        return
    if not aggregate or any(row is None or set(row) != set(aggregate) for row in rows):
        raise ValueError(f"Incomplete per-question {label}")
    complete = [row for row in rows if row is not None]
    for key, value in aggregate.items():
        if not isclose(value, fmean(row[key] for row in complete), abs_tol=1e-12):
            raise ValueError(f"Aggregate {label} does not match per-question values")


def _aligned_scalar(aggregate: float | None, values: Sequence[float | None], label: str) -> None:
    if aggregate is None:
        if any(value is not None for value in values):
            raise ValueError(f"Missing aggregate {label}")
        return
    if any(value is None for value in values):
        raise ValueError(f"Incomplete per-question {label}")
    observed = [value for value in values if value is not None]
    if not isclose(aggregate, fmean(observed), abs_tol=1e-12):
        raise ValueError(f"Aggregate {label} does not match per-question values")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkQuestion(Record):
    id: Nonblank
    question: Nonblank
    expected_answer: str = ""
    relevant_document_ids: tuple[Nonblank, ...]
    relevance_grades: dict[Nonblank, PositiveGrade] | None = None

    @model_validator(mode="after")
    def validate_relevance(self) -> Self:
        ids = self.relevant_document_ids
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("relevant_document_ids must be nonempty and distinct")
        if self.expected_answer and not self.expected_answer.strip():
            raise ValueError("expected_answer must be empty or contain non-whitespace text")
        if self.relevance_grades is not None and (
            set(self.relevance_grades) != set(ids)
            or any(
                isinstance(grade, bool) or grade <= 0 for grade in self.relevance_grades.values()
            )
        ):
            raise ValueError("relevance_grades must be positive and match relevant_document_ids")
        return self

    def grades(self) -> dict[str, int]:
        if self.relevance_grades is None:
            return dict.fromkeys(self.relevant_document_ids, 1)
        return dict(self.relevance_grades)


class Benchmark(Record):
    schema_version: SchemaVersion = 1
    questions: tuple[BenchmarkQuestion, ...]

    @model_validator(mode="after")
    def validate_questions(self) -> Self:
        if not self.questions or len({question.id for question in self.questions}) != len(
            self.questions
        ):
            raise ValueError("Benchmark must contain questions with unique IDs")
        return self


class DocumentHit(Record):
    document_id: str
    chunk_id: str
    score: float = Field(allow_inf_nan=False)


class QuestionResult(Record):
    id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    retrieved_documents: tuple[DocumentHit, ...]
    recall_at_k: dict[int, UnitFloat]
    reciprocal_rank: UnitFloat
    ndcg_at_k: dict[int, UnitFloat] | None = None
    precision_at_k: dict[int, UnitFloat] | None = None
    average_precision: UnitFloat | None = None
    retrieval_ms: NonnegativeFloat = 0.0
    generation_ms: NonnegativeFloat | None = None
    judge_ms: NonnegativeFloat | None = None
    answer: GeneratedAnswer | None = None
    answer_metrics: dict[str, UnitFloat] | None = None
    context_chunk_ids: tuple[str, ...] = ()
    judge: JudgeResult | None = None


class Timings(Record):
    ingestion_ms: NonnegativeFloat
    index_ms: NonnegativeFloat
    retrieval_p50_ms: NonnegativeFloat
    retrieval_p95_ms: NonnegativeFloat
    generation_p95_ms: NonnegativeFloat | None = None
    judge_p95_ms: NonnegativeFloat | None = None


class EvaluationResult(Record):
    schema_version: Literal[2] = 2
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: RunConfig
    corpus_sha256: str
    benchmark_sha256: str
    document_count: int
    chunk_count: int
    question_count: int
    skipped_empty_files: int
    retriever: dict[str, str | int]
    versions: dict[str, str]
    recall_at_k: dict[int, UnitFloat]
    mrr: UnitFloat
    ndcg_at_k: dict[int, UnitFloat] | None = None
    precision_at_k: dict[int, UnitFloat] | None = None
    mean_average_precision: UnitFloat | None = None
    multi_chunk_documents: int | None = Field(default=None, ge=0)
    questions: tuple[QuestionResult, ...]
    timings: Timings
    environment: dict[str, str]
    answer_metrics: dict[str, UnitFloat] | None = None
    judge_metrics: dict[str, UnitFloat] | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: NonnegativeFloat | None = None

    @model_validator(mode="after")
    def coherent_report(self) -> Self:
        if not self.questions or self.question_count != len(self.questions):
            raise ValueError("question_count must match nonempty questions")
        if len({q.id for q in self.questions}) != self.question_count:
            raise ValueError("Report question IDs must be unique")
        if (
            self.multi_chunk_documents is not None
            and self.multi_chunk_documents > self.document_count
        ):
            raise ValueError("multi_chunk_documents cannot exceed document_count")
        if not isclose(self.mrr, fmean(q.reciprocal_rank for q in self.questions), abs_tol=1e-12):
            raise ValueError("Aggregate MRR does not match per-question values")
        _aligned_means(self.recall_at_k, [q.recall_at_k for q in self.questions], "recall")
        _aligned_means(self.ndcg_at_k, [q.ndcg_at_k for q in self.questions], "nDCG")
        _aligned_means(self.precision_at_k, [q.precision_at_k for q in self.questions], "precision")
        _aligned_scalar(
            self.mean_average_precision,
            [q.average_precision for q in self.questions],
            "mean average precision",
        )
        _aligned_means(
            self.answer_metrics,
            [question.answer_metrics for question in self.questions],
            "answer_metrics",
        )
        _aligned_means(
            self.judge_metrics,
            [
                {
                    "judge_correctness": question.judge.scores.correctness / 4,
                    "judge_faithfulness": question.judge.scores.faithfulness / 4,
                }
                if question.judge
                else None
                for question in self.questions
            ],
            "judge_metrics",
        )
        return self
