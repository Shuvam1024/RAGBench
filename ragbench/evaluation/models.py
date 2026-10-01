"""Versioned benchmark inputs and inspectable evaluation outputs."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragbench.config import Nonblank, RunConfig, SchemaVersion


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
        if not self.questions or len({question.id for question in self.questions}) != len(self.questions):
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
    recall_at_k: dict[int, float]
    reciprocal_rank: float


class EvaluationResult(Record):
    schema_version: SchemaVersion = 1
    config: RunConfig
    corpus_sha256: str
    benchmark_sha256: str
    document_count: int
    chunk_count: int
    question_count: int
    skipped_empty_files: int
    retriever: dict[str, str | int]
    versions: dict[str, str]
    recall_at_k: dict[int, float]
    mrr: float
    questions: tuple[QuestionResult, ...]
