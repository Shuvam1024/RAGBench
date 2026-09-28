"""Orchestrate retrieval evaluation without terminal or output-file dependencies."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from statistics import fmean

from ragbench.config import BM25Config, DenseConfig, RetrieverConfig, RunConfig
from ragbench.evaluation.models import Benchmark, DocumentHit, EvaluationResult, QuestionResult
from ragbench.evaluation.retrieval import recall_at_k, reciprocal_rank
from ragbench.ingestion.chunker import chunk_documents
from ragbench.ingestion.loader import load_documents
from ragbench.retrieval.base import Retriever
from ragbench.retrieval.bm25 import BM25Retriever


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_benchmark(path: Path, document_ids: set[str]) -> Benchmark:
    try:
        benchmark = Benchmark.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot load benchmark {path}: {exc}") from exc
    for question in benchmark.questions:
        missing = set(question.relevant_document_ids) - document_ids
        if missing:
            raise ValueError(f"Question {question.id!r} references unknown documents: {sorted(missing)}")
    return benchmark


def build_retriever(config: RetrieverConfig) -> Retriever:
    if isinstance(config, BM25Config):
        return BM25Retriever(config)
    try:
        from ragbench.retrieval.dense import DenseRetriever
    except ImportError as exc:
        raise ValueError('Dense retrieval requires: python -m pip install -e ".[dense]"') from exc
    if isinstance(config, DenseConfig):
        return DenseRetriever(config)
    raise ValueError(f"Unsupported retrieval configuration: {config}")


def dependency_versions(dense: bool) -> dict[str, str]:
    names = ["ragbench", "pydantic", "PyYAML", "rank-bm25", "numpy", "typer"]
    if dense:
        names += ["sentence-transformers", "faiss-cpu", "torch", "transformers", "huggingface-hub"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def evaluate(config: RunConfig, retriever: Retriever | None = None) -> EvaluationResult:
    """Validate labels before indexing; search with question text alone."""
    corpus = load_documents(config.dataset.documents_path)
    benchmark = load_benchmark(config.dataset.benchmark_path, {doc.id for doc in corpus.documents})
    chunks = chunk_documents(corpus.documents, config.chunking)
    retriever = retriever or build_retriever(config.retrieval)
    retriever.index(chunks)
    results: list[QuestionResult] = []
    for question in benchmark.questions:
        hits: dict[str, DocumentHit] = {}
        for hit in retriever.retrieve(question.question, len(chunks)):
            hits.setdefault(hit.chunk.document_id, DocumentHit(
                document_id=hit.chunk.document_id, chunk_id=hit.chunk.id, score=hit.score,
            ))
        ranking = list(hits)
        results.append(QuestionResult(
            id=question.id, question=question.question, relevant_document_ids=question.relevant_document_ids,
            retrieved_documents=tuple(hits.values()),
            recall_at_k={k: recall_at_k(ranking, question.relevant_document_ids, k)
                         for k in config.evaluation.recall_at_k},
            reciprocal_rank=reciprocal_rank(ranking, question.relevant_document_ids),
        ))
    return EvaluationResult(
        config=config, corpus_sha256=fingerprint([(doc.id, doc.content_sha256) for doc in corpus.documents]),
        benchmark_sha256=fingerprint(benchmark.model_dump(mode="json")),
        document_count=len(corpus.documents), chunk_count=len(chunks), question_count=len(results),
        skipped_empty_files=corpus.skipped_empty_files, retriever=retriever.metadata,
        versions=dependency_versions(not isinstance(config.retrieval, BM25Config)),
        recall_at_k={k: fmean(result.recall_at_k[k] for result in results)
                     for k in config.evaluation.recall_at_k},
        mrr=fmean(result.reciprocal_rank for result in results), questions=tuple(results),
    )
