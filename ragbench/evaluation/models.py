"""Versioned benchmark inputs and inspectable evaluation outputs."""

from datetime import datetime, timezone
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragbench.config import Nonblank, NonnegativeFloat, RunConfig, SchemaVersion, UnitFloat
from ragbench.generation.models import GeneratedAnswer, JudgeResult


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkQuestion(Record):
    id: Nonblank
    question: Nonblank
    expected_answer: Nonblank
    relevant_document_ids: tuple[Nonblank, ...]

    @model_validator(mode="after")
    def validate_relevance(self) -> Self:
        ids = self.relevant_document_ids
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("relevant_document_ids must be nonempty and distinct")
        return self


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
        from math import isclose
        from statistics import fmean

        if not self.questions or self.question_count != len(self.questions):
            raise ValueError("question_count must match nonempty questions")
        if len({q.id for q in self.questions}) != self.question_count:
            raise ValueError("Report question IDs must be unique")
        if not isclose(self.mrr, fmean(q.reciprocal_rank for q in self.questions), abs_tol=1e-12):
            raise ValueError("Aggregate MRR does not match per-question values")
        for k, value in self.recall_at_k.items():
            if any(k not in q.recall_at_k for q in self.questions):
                raise ValueError("Recall cutoff missing from a question")
            if not isclose(value, fmean(q.recall_at_k[k] for q in self.questions), abs_tol=1e-12):
                raise ValueError("Aggregate recall does not match per-question values")
        for field in ("answer_metrics", "judge_metrics"):
            aggregate = getattr(self, field)
            if field == "answer_metrics":
                individual = [q.answer_metrics for q in self.questions]
            else:
                individual = [
                    {
                        "judge_correctness": q.judge.scores.correctness / 4,
                        "judge_faithfulness": q.judge.scores.faithfulness / 4,
                    }
                    if q.judge
                    else None
                    for q in self.questions
                ]
            if aggregate is None:
                if any(item is not None for item in individual):
                    raise ValueError(f"Missing aggregate {field}")
            elif not aggregate or any(
                item is None or set(item) != set(aggregate) for item in individual
            ):
                raise ValueError(f"Incomplete per-question {field}")
            elif any(
                not isclose(value, fmean(item[name] for item in individual), abs_tol=1e-12)
                for name, value in aggregate.items()
            ):
                raise ValueError(f"Aggregate {field} does not match per-question values")
        return self
