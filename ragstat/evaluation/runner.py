"""Orchestrate retrieval evaluation without terminal or output-file dependencies."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean
from time import perf_counter

from ragstat.config import BM25Config, DenseConfig, HybridConfig, RetrieverConfig, RunConfig
from ragstat.evaluation.answers import score_answer
from ragstat.evaluation.judge import LLMJudge
from ragstat.evaluation.models import (
    Benchmark,
    DocumentHit,
    EvaluationResult,
    QuestionResult,
    Timings,
)
from ragstat.evaluation.retrieval import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragstat.evaluation.timing import percentile
from ragstat.generation.providers import ContextSource, Generator, build_generator
from ragstat.ingestion.chunker import Chunk, chunk_documents
from ragstat.ingestion.loader import Document, load_documents
from ragstat.retrieval.base import Retriever
from ragstat.retrieval.bm25 import BM25Retriever
from ragstat.retrieval.rerank import PairScorer, rank_candidates, select_candidates


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
        from ragstat.retrieval.dense import DenseRetriever
    except ImportError as exc:
        raise ValueError('Dense retrieval requires: python -m pip install -e ".[dense]"') from exc
    if isinstance(config, DenseConfig):
        return DenseRetriever(config)
    if isinstance(config, HybridConfig):
        from ragstat.retrieval.hybrid import HybridRetriever

        return HybridRetriever(
            BM25Retriever(config.bm25),
            DenseRetriever(config.dense),
            config.dense_weight,
            fusion=config.fusion,
            rrf_k=config.rrf_k,
        )
    raise ValueError(f"Unsupported retrieval configuration: {config}")


def dependency_versions(dense: bool) -> dict[str, str]:
    names = [
        "ragstat",
        "pydantic",
        "PyYAML",
        "rank-bm25",
        "snowballstemmer",
        "numpy",
        "typer",
        "httpx",
    ]
    if dense:
        names += ["sentence-transformers", "faiss-cpu", "torch", "transformers", "huggingface-hub"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _total_cost(costs: list[float | None]) -> float | None:
    if not costs:
        return None
    total = 0.0
    for cost in costs:
        if cost is None:
            return None
        total += cost
    return total


def pytorch_threads() -> str:
    """Record the live thread count, or that PyTorch is not installed."""
    if importlib.util.find_spec("torch") is None:
        return "not-installed"
    import torch

    return str(torch.get_num_threads())


def _mean_cutoffs(rows: list[dict[int, float] | None], label: str) -> dict[int, float]:
    if any(row is None for row in rows):
        raise ValueError(f"Missing per-question {label}")
    complete = [row for row in rows if row is not None]
    cutoffs = complete[0]
    if any(set(row) != set(cutoffs) for row in complete):
        raise ValueError(f"Inconsistent per-question {label}")
    return {cutoff: fmean(row[cutoff] for row in complete) for cutoff in cutoffs}


def _answer_metrics(results: list[QuestionResult]) -> dict[str, float]:
    first = results[0].answer_metrics
    if first is None:
        raise ValueError("Generation produced incomplete answer metrics")
    aggregated: dict[str, float] = {}
    for name in first:
        values: list[float] = []
        for question in results:
            metrics = question.answer_metrics
            if metrics is None or name not in metrics:
                raise ValueError("Generation produced incomplete answer metrics")
            values.append(metrics[name])
        aggregated[name] = fmean(values)
    return aggregated


def _judge_metrics(results: list[QuestionResult]) -> dict[str, float]:
    correctness: list[float] = []
    faithfulness: list[float] = []
    for question in results:
        judged = question.judge
        if judged is None:
            raise ValueError("Judge produced incomplete scores")
        correctness.append(judged.scores.correctness / 4)
        faithfulness.append(judged.scores.faithfulness / 4)
    return {
        "judge_correctness": fmean(correctness),
        "judge_faithfulness": fmean(faithfulness),
    }


def relativize_result(result: EvaluationResult, root: Path) -> EvaluationResult:
    """Rewrite report paths that live under ``root`` as POSIX relative paths."""

    def relative(path: Path | None) -> Path | None:
        if path is None:
            return None
        try:
            return Path(path.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            return path

    config = result.config
    return result.model_copy(
        update={
            "config": config.model_copy(
                update={
                    "dataset": config.dataset.model_copy(
                        update={
                            "documents_path": relative(config.dataset.documents_path),
                            "benchmark_path": relative(config.dataset.benchmark_path),
                        }
                    ),
                    "output": config.output.model_copy(
                        update={"json_path": relative(config.output.json_path)}
                    ),
                    "storage": config.storage.model_copy(
                        update={"sqlite_path": relative(config.storage.sqlite_path)}
                    ),
                }
            )
        }
    )


def _validate_depths(config: RunConfig, depths: tuple[int, ...]) -> tuple[int, ...]:
    if config.rerank is None:
        raise ValueError("rerank settings are required")
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("rerank depths must be distinct positive integers")
    for depth in depths:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("rerank depths must be distinct positive integers")
        if depth > config.rerank.candidate_k:
            raise ValueError("rerank depth cannot exceed candidate_k")
    if max(depths) != config.rerank.candidate_k:
        raise ValueError("the widest rerank depth must equal candidate_k")
    return tuple(sorted(depths))


def _uses_model(config: RunConfig) -> bool:
    return not isinstance(config.retrieval, BM25Config) or config.rerank is not None


def _assemble(
    *,
    config: RunConfig,
    corpus_documents: Sequence[Document],
    benchmark: Benchmark,
    chunks: Sequence[Chunk],
    multi_chunk_documents: int,
    skipped_empty_files: int,
    retriever: dict[str, str | int],
    results: list[QuestionResult],
    ingestion_ms: float,
    index_ms: float,
    rerank_times: list[float] | None,
    pipeline_times: list[float] | None,
    uses_model: bool,
) -> EvaluationResult:
    responses = [question.answer for question in results if question.answer is not None] + [
        question.judge.response for question in results if question.judge is not None
    ]
    usage = [response.usage for response in responses if response.usage is not None]
    costs = [response.estimated_cost_usd for response in responses]
    candidate_values = [question.candidate_recall_at_100 for question in results]
    return EvaluationResult(
        config=config,
        corpus_sha256=fingerprint([(doc.id, doc.content_sha256) for doc in corpus_documents]),
        benchmark_sha256=fingerprint(benchmark.model_dump(mode="json", exclude_none=True)),
        document_count=len(corpus_documents),
        chunk_count=len(chunks),
        multi_chunk_documents=multi_chunk_documents,
        question_count=len(results),
        skipped_empty_files=skipped_empty_files,
        retriever=retriever,
        versions=dependency_versions(uses_model),
        recall_at_k={
            cutoff: fmean(result.recall_at_k[cutoff] for result in results)
            for cutoff in config.evaluation.recall_at_k
        },
        ndcg_at_k=_mean_cutoffs([result.ndcg_at_k for result in results], "nDCG"),
        precision_at_k=_mean_cutoffs([result.precision_at_k for result in results], "precision"),
        mean_average_precision=fmean(
            result.average_precision for result in results if result.average_precision is not None
        ),
        candidate_recall_at_100=(
            fmean(value for value in candidate_values if value is not None)
            if any(value is not None for value in candidate_values)
            else None
        ),
        mrr=fmean(result.reciprocal_rank for result in results),
        questions=tuple(results),
        timings=Timings(
            ingestion_ms=ingestion_ms,
            index_ms=index_ms,
            retrieval_p50_ms=percentile([question.retrieval_ms for question in results], 0.5),
            retrieval_p95_ms=percentile([question.retrieval_ms for question in results], 0.95),
            rerank_p50_ms=percentile(rerank_times, 0.5) if rerank_times else None,
            rerank_p95_ms=percentile(rerank_times, 0.95) if rerank_times else None,
            pipeline_p50_ms=percentile(pipeline_times, 0.5) if pipeline_times else None,
            pipeline_p95_ms=percentile(pipeline_times, 0.95) if pipeline_times else None,
            generation_p95_ms=percentile(
                [
                    question.generation_ms
                    for question in results
                    if question.generation_ms is not None
                ],
                0.95,
            )
            if config.generation
            else None,
            judge_p95_ms=percentile(
                [question.judge_ms for question in results if question.judge_ms is not None],
                0.95,
            )
            if config.judge
            else None,
        ),
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": str(os.cpu_count()),
            "pytorch_threads": pytorch_threads(),
        },
        answer_metrics=_answer_metrics(results) if config.generation else None,
        judge_metrics=_judge_metrics(results) if config.judge else None,
        input_tokens=sum(item.input_tokens for item in usage) if usage else None,
        output_tokens=sum(item.output_tokens for item in usage) if usage else None,
        estimated_cost_usd=_total_cost(costs),
    )


def evaluate_candidate_depths(
    config: RunConfig,
    depths: tuple[int, ...],
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    judge: LLMJudge | None = None,
    reranker: PairScorer | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[int, EvaluationResult]:
    """Score the widest candidate set once and derive each smaller depth from it.

    ``retrieval_ms`` is the full-ranking first stage, which scores every chunk.
    Min-max hybrid has to do that before it can name a top-K set. ``rerank_ms``
    is only the cross-encoder. ``pipeline_ms`` adds those two. It is the
    end-to-end cost of the reranked top-K list, and the first-stage part of
    that sum is still the full ranking.
    """
    rerank_config = config.rerank
    ordered_depths = _validate_depths(config, depths)
    assert rerank_config is not None
    if config.generation is not None and len(ordered_depths) != 1:
        raise ValueError("generation is only supported for one rerank depth")
    started = perf_counter()
    corpus = load_documents(config.dataset.documents_path)
    benchmark = load_benchmark(config.dataset.benchmark_path, {doc.id for doc in corpus.documents})
    if config.generation is not None:
        missing_answers = [
            item.id for item in benchmark.questions if not item.expected_answer.strip()
        ]
        if missing_answers:
            raise ValueError(
                "Generation requires expected_answer; missing for " + ", ".join(missing_answers[:5])
            )
        generator = generator or build_generator(config.generation)
    if config.judge is not None:
        judge = judge or LLMJudge(config.judge)
    chunks = chunk_documents(corpus.documents, config.chunking)
    ingestion_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    retriever = retriever or build_retriever(config.retrieval)
    retriever.index(chunks)
    if reranker is None:
        from ragstat.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(rerank_config)
    index_ms = (perf_counter() - started) * 1000
    by_depth: dict[int, list[QuestionResult]] = {depth: [] for depth in ordered_depths}
    truncated_totals = dict.fromkeys(ordered_depths, 0)
    for index, question in enumerate(benchmark.questions, start=1):
        started = perf_counter()
        chunk_hits = retriever.retrieve(question.question, len(chunks))
        retrieval_ms = (perf_counter() - started) * 1000
        first_stage = [item.document_id for item in select_candidates(chunk_hits, candidate_k=None)]
        candidate_recall = recall_at_k(first_stage, question.relevant_document_ids, 100)
        pool = select_candidates(chunk_hits, rerank_config.candidate_k)
        before = reranker.truncated_pairs
        started = perf_counter()
        scored = reranker.score([(question.question, item.text) for item in pool]) if pool else []
        widest_rerank_ms = (perf_counter() - started) * 1000
        truncated_totals[ordered_depths[-1]] += reranker.truncated_pairs - before
        for depth in ordered_depths:
            # The clock covers the widest pool. Smaller depths reuse those scores.
            rerank_ms = widest_rerank_ms if depth == ordered_depths[-1] else None
            pipeline_ms = retrieval_ms + widest_rerank_ms if rerank_ms is not None else None
            ordered = rank_candidates(pool[:depth], scored[:depth])
            ranking = [item.document_id for item, _score in ordered]
            grades = question.grades()
            recall = {
                cutoff: recall_at_k(ranking, question.relevant_document_ids, cutoff)
                for cutoff in config.evaluation.recall_at_k
            }
            ndcg = {
                cutoff: ndcg_at_k(ranking, grades, cutoff)
                for cutoff in config.evaluation.ndcg_cutoffs()
            }
            precision = {
                cutoff: precision_at_k(ranking, question.relevant_document_ids, cutoff)
                for cutoff in config.evaluation.precision_cutoffs()
            }
            retrieved = tuple(
                DocumentHit(document_id=item.document_id, chunk_id=item.chunk_id, score=score)
                for item, score in ordered
            )
            if config.evaluation.stored_hits is not None:
                retrieved = retrieved[: config.evaluation.stored_hits]
            answer = None
            answer_metrics = None
            judged = None
            generation_ms = judge_ms = None
            context: list[ContextSource] = []
            if config.generation is not None:
                chunks_by_id = {hit.chunk.id: hit.chunk for hit in chunk_hits}
                budget = config.generation.max_context_chars
                for item, _score in ordered:
                    if len(context) >= config.generation.context_k or budget <= 0:
                        break
                    text = chunks_by_id[item.chunk_id].text[:budget]
                    budget -= len(text)
                    context.append(
                        ContextSource(
                            document_id=item.document_id,
                            chunk_id=item.chunk_id,
                            text=text,
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
            by_depth[depth].append(
                QuestionResult(
                    id=question.id,
                    question=question.question,
                    relevant_document_ids=question.relevant_document_ids,
                    retrieved_documents=retrieved,
                    recall_at_k=recall,
                    ndcg_at_k=ndcg,
                    precision_at_k=precision,
                    average_precision=average_precision(ranking, question.relevant_document_ids),
                    reciprocal_rank=reciprocal_rank(ranking, question.relevant_document_ids),
                    candidate_recall_at_100=candidate_recall,
                    retrieval_ms=retrieval_ms,
                    rerank_ms=rerank_ms,
                    pipeline_ms=pipeline_ms,
                    answer=answer,
                    answer_metrics=answer_metrics,
                    generation_ms=generation_ms,
                    judge_ms=judge_ms,
                    context_chunk_ids=tuple(source.chunk_id for source in context),
                    judge=judged,
                )
            )
        if progress is not None:
            progress(index, len(benchmark.questions))
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk.document_id] = chunk_counts.get(chunk.document_id, 0) + 1
    reports: dict[int, EvaluationResult] = {}
    for depth in ordered_depths:
        depth_config = config.model_copy(
            update={"rerank": rerank_config.model_copy(update={"candidate_k": depth})}
        )
        results = by_depth[depth]
        details: dict[str, str | int] = {**retriever.metadata, **reranker.metadata}
        details["candidate_k"] = depth
        if depth == ordered_depths[-1]:
            details["rerank_truncated_pairs"] = truncated_totals[depth]
        else:
            details.pop("rerank_truncated_pairs", None)
        rerank_times = [
            question.rerank_ms for question in results if question.rerank_ms is not None
        ]
        pipeline_times = [
            question.pipeline_ms for question in results if question.pipeline_ms is not None
        ]
        reports[depth] = _assemble(
            config=depth_config,
            corpus_documents=corpus.documents,
            benchmark=benchmark,
            chunks=chunks,
            multi_chunk_documents=sum(count > 1 for count in chunk_counts.values()),
            skipped_empty_files=corpus.skipped_empty_files,
            retriever=details,
            results=results,
            ingestion_ms=ingestion_ms,
            index_ms=index_ms,
            rerank_times=rerank_times,
            pipeline_times=pipeline_times,
            uses_model=_uses_model(config),
        )
    return reports


def evaluate(
    config: RunConfig,
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    judge: LLMJudge | None = None,
    reranker: PairScorer | None = None,
) -> EvaluationResult:
    """Validate labels before indexing; search with question text alone."""
    if reranker is not None and config.rerank is None:
        raise ValueError("reranker requires rerank settings")
    if config.rerank is not None:
        return evaluate_candidate_depths(
            config,
            (config.rerank.candidate_k,),
            retriever=retriever,
            generator=generator,
            judge=judge,
            reranker=reranker,
        )[config.rerank.candidate_k]
    started = perf_counter()
    corpus = load_documents(config.dataset.documents_path)
    benchmark = load_benchmark(config.dataset.benchmark_path, {doc.id for doc in corpus.documents})
    if config.generation is not None:
        missing_answers = [
            item.id for item in benchmark.questions if not item.expected_answer.strip()
        ]
        if missing_answers:
            raise ValueError(
                "Generation requires expected_answer; missing for " + ", ".join(missing_answers[:5])
            )
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
        grades = question.grades()
        recall = {
            cutoff: recall_at_k(ranking, question.relevant_document_ids, cutoff)
            for cutoff in config.evaluation.recall_at_k
        }
        ndcg = {
            cutoff: ndcg_at_k(ranking, grades, cutoff)
            for cutoff in config.evaluation.ndcg_cutoffs()
        }
        precision = {
            cutoff: precision_at_k(ranking, question.relevant_document_ids, cutoff)
            for cutoff in config.evaluation.precision_cutoffs()
        }
        retrieved = tuple(hits.values())
        if config.evaluation.stored_hits is not None:
            retrieved = retrieved[: config.evaluation.stored_hits]
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
                retrieved_documents=retrieved,
                recall_at_k=recall,
                ndcg_at_k=ndcg,
                precision_at_k=precision,
                average_precision=average_precision(ranking, question.relevant_document_ids),
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
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk.document_id] = chunk_counts.get(chunk.document_id, 0) + 1
    return EvaluationResult(
        config=config,
        corpus_sha256=fingerprint([(doc.id, doc.content_sha256) for doc in corpus.documents]),
        benchmark_sha256=fingerprint(benchmark.model_dump(mode="json", exclude_none=True)),
        document_count=len(corpus.documents),
        chunk_count=len(chunks),
        multi_chunk_documents=sum(count > 1 for count in chunk_counts.values()),
        question_count=len(results),
        skipped_empty_files=corpus.skipped_empty_files,
        retriever=retriever.metadata,
        versions=dependency_versions(not isinstance(config.retrieval, BM25Config)),
        recall_at_k={
            k: fmean(result.recall_at_k[k] for result in results)
            for k in config.evaluation.recall_at_k
        },
        ndcg_at_k=_mean_cutoffs([result.ndcg_at_k for result in results], "nDCG"),
        precision_at_k=_mean_cutoffs([result.precision_at_k for result in results], "precision"),
        mean_average_precision=fmean(
            result.average_precision for result in results if result.average_precision is not None
        ),
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
            "pytorch_threads": pytorch_threads(),
        },
        answer_metrics=_answer_metrics(results) if config.generation else None,
        judge_metrics=_judge_metrics(results) if config.judge else None,
        input_tokens=sum(item.input_tokens for item in usage) if usage else None,
        output_tokens=sum(item.output_tokens for item in usage) if usage else None,
        estimated_cost_usd=_total_cost(costs),
    )
