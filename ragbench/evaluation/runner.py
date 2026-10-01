"""Orchestrate retrieval evaluation without terminal or output-file dependencies."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from statistics import fmean
from time import perf_counter

from ragbench.config import BM25Config, DenseConfig, HybridConfig, RetrieverConfig, RunConfig
from ragbench.evaluation.answers import score_answer
from ragbench.evaluation.judge import LLMJudge
from ragbench.evaluation.models import (
    Benchmark,
    DocumentHit,
    EvaluationResult,
    QuestionResult,
    Timings,
)
from ragbench.evaluation.retrieval import recall_at_k, reciprocal_rank
from ragbench.evaluation.timing import percentile
from ragbench.generation.providers import ContextSource, Generator, build_generator
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
            raise ValueError(
                f"Question {question.id!r} references unknown documents: {sorted(missing)}"
            )
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
    if isinstance(config, HybridConfig):
        from ragbench.retrieval.hybrid import HybridRetriever

        return HybridRetriever(
            BM25Retriever(config.bm25), DenseRetriever(config.dense), config.dense_weight
        )
    raise ValueError(f"Unsupported retrieval configuration: {config}")


def dependency_versions(dense: bool) -> dict[str, str]:
    names = ["ragbench", "pydantic", "PyYAML", "rank-bm25", "numpy", "typer", "httpx"]
    if dense:
        names += ["sentence-transformers", "faiss-cpu", "torch", "transformers", "huggingface-hub"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def evaluate(
    config: RunConfig,
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    judge: LLMJudge | None = None,
) -> EvaluationResult:
    """Validate labels before indexing; search with question text alone."""
    started = perf_counter()
    corpus = load_documents(config.dataset.documents_path)
    benchmark = load_benchmark(config.dataset.benchmark_path, {doc.id for doc in corpus.documents})
    if config.generation is not None:
        generator = generator or build_generator(config.generation)
    if config.judge is not None:
        judge = judge or LLMJudge(config.judge)
    chunks = chunk_documents(corpus.documents, config.chunking)
    ingestion_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    retriever = retriever or build_retriever(config.retrieval)
    retriever.index(chunks)
    index_ms = (perf_counter() - started) * 1000
    results: list[QuestionResult] = []
    for question in benchmark.questions:
        started = perf_counter()
        chunk_hits = retriever.retrieve(question.question, len(chunks))
        retrieval_ms = (perf_counter() - started) * 1000
        hits: dict[str, DocumentHit] = {}
        for hit in chunk_hits:
            hits.setdefault(
                hit.chunk.document_id,
                DocumentHit(
                    document_id=hit.chunk.document_id,
                    chunk_id=hit.chunk.id,
                    score=hit.score,
                ),
            )
        ranking = list(hits)
        answer = None
        answer_metrics = None
        judged = None
        generation_ms = judge_ms = None
        context: list[ContextSource] = []
        if config.generation is not None:
            budget = config.generation.max_context_chars
            seen: set[str] = set()
            for hit in chunk_hits:
                if hit.chunk.document_id in seen:
                    continue
                if len(context) >= config.generation.context_k or budget <= 0:
                    break
                seen.add(hit.chunk.document_id)
                text = hit.chunk.text[:budget]
                budget -= len(text)
                context.append(
                    ContextSource(
                        document_id=hit.chunk.document_id, chunk_id=hit.chunk.id, text=text
                    )
                )
            assert generator is not None
            started = perf_counter()
            answer = generator.generate(question.question, context)
            generation_ms = (perf_counter() - started) * 1000
            answer_metrics = score_answer(answer.text, question.expected_answer, context)
            if config.judge is not None:
                assert judge is not None
                started = perf_counter()
                judged = judge.assess(
                    question.question, question.expected_answer, answer.text, context
                )
                judge_ms = (perf_counter() - started) * 1000
        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                relevant_document_ids=question.relevant_document_ids,
                retrieved_documents=tuple(hits.values()),
                recall_at_k={
                    k: recall_at_k(ranking, question.relevant_document_ids, k)
                    for k in config.evaluation.recall_at_k
                },
                reciprocal_rank=reciprocal_rank(ranking, question.relevant_document_ids),
                retrieval_ms=retrieval_ms,
                answer=answer,
                answer_metrics=answer_metrics,
                generation_ms=generation_ms,
                judge_ms=judge_ms,
                context_chunk_ids=tuple(source.chunk_id for source in context),
                judge=judged,
            )
        )
    responses = [q.answer for q in results if q.answer is not None] + [
        q.judge.response for q in results if q.judge is not None
    ]
    usage = [response.usage for response in responses if response.usage is not None]
    costs = [response.estimated_cost_usd for response in responses]
    return EvaluationResult(
        config=config,
        corpus_sha256=fingerprint([(doc.id, doc.content_sha256) for doc in corpus.documents]),
        benchmark_sha256=fingerprint(benchmark.model_dump(mode="json")),
        document_count=len(corpus.documents),
        chunk_count=len(chunks),
        question_count=len(results),
        skipped_empty_files=corpus.skipped_empty_files,
        retriever=retriever.metadata,
        versions=dependency_versions(not isinstance(config.retrieval, BM25Config)),
        recall_at_k={
            k: fmean(result.recall_at_k[k] for result in results)
            for k in config.evaluation.recall_at_k
        },
        mrr=fmean(result.reciprocal_rank for result in results),
        questions=tuple(results),
        timings=Timings(
            ingestion_ms=ingestion_ms,
            index_ms=index_ms,
            retrieval_p50_ms=percentile([q.retrieval_ms for q in results], 0.5),
            retrieval_p95_ms=percentile([q.retrieval_ms for q in results], 0.95),
            generation_p95_ms=percentile(
                [q.generation_ms for q in results if q.generation_ms is not None], 0.95
            )
            if config.generation
            else None,
            judge_p95_ms=percentile([q.judge_ms for q in results if q.judge_ms is not None], 0.95)
            if config.judge
            else None,
        ),
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": str(os.cpu_count()),
            "pytorch_threads": "1",
        },
        answer_metrics={
            name: fmean(q.answer_metrics[name] for q in results)
            for name in results[0].answer_metrics
        }
        if config.generation
        else None,
        judge_metrics={
            "judge_correctness": fmean(q.judge.scores.correctness / 4 for q in results),
            "judge_faithfulness": fmean(q.judge.scores.faithfulness / 4 for q in results),
        }
        if config.judge
        else None,
        input_tokens=sum(item.input_tokens for item in usage) if usage else None,
        output_tokens=sum(item.output_tokens for item in usage) if usage else None,
        estimated_cost_usd=sum(costs)
        if costs and all(cost is not None for cost in costs)
        else None,
    )
