"""Choose a SciFact verdict policy on the train claims.

The dev split is not scored. The first stage is the frozen document-level BM25
index. The NLI forward pass covers the widest ``doc_k`` once. The winner
maximizes train macro-F1. Ties prefer higher mean sentence F1, then smaller
``doc_k``, then smaller ``sentence_k``, then higher ``min_confidence``.
"""

import json
from pathlib import Path
from time import perf_counter

from ragstat.datasets.beir import _read_qrels, _read_queries
from ragstat.datasets.scifact_claims import (
    compare_abstracts_to_beir_bodies,
    load_beir_corpus_fields,
    prepare_scifact_release,
    require_scifact_beir_mapping,
)
from ragstat.evaluation.nli import CrossEncoderNli
from ragstat.evaluation.runner import pytorch_threads
from ragstat.evaluation.timing import percentile
from ragstat.evaluation.verdict import (
    TIE_BREAK,
    assert_train_selection,
    build_sentence_jobs,
    choose_policy,
    evaluate_predictions,
    iter_policies,
    load_verdict_experiment,
    majority_verdict,
    predict_baseline,
    predictions_for_policy,
    retrieve_top_documents,
    score_sentence_jobs,
)
from ragstat.ingestion.loader import load_documents

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scifact-verdict-train.yaml"


def _environment() -> dict[str, str]:
    import os
    import platform

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
        "pytorch_threads": pytorch_threads(),
    }


def _row(policy: object, report: object) -> dict[str, object]:
    assert hasattr(policy, "doc_k") and hasattr(report, "macro_f1")
    return {
        "doc_k": policy.doc_k,
        "sentence_k": policy.sentence_k,
        "min_confidence": policy.min_confidence,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "sentence_precision": report.sentence_precision,
        "sentence_recall": report.sentence_recall,
        "sentence_f1": report.sentence_f1,
        "micro_sentence_precision": report.micro_sentence_precision,
        "micro_sentence_recall": report.micro_sentence_recall,
        "micro_sentence_f1": report.micro_sentence_f1,
    }


def main() -> None:
    experiment = load_verdict_experiment(CONFIG)
    assert_train_selection(experiment.split)
    retrieval = experiment.retrieval
    if (retrieval.k1, retrieval.b, retrieval.tokenizer, retrieval.unit) != (
        0.9,
        0.4,
        "stem",
        "document",
    ):
        raise SystemExit("refusing to select: retrieval is not the frozen document-level BM25")
    release = prepare_scifact_release(experiment.claims_cache)
    raw = experiment.beir_raw
    mapping = require_scifact_beir_mapping(
        release.train,
        release.dev,
        release.test,
        _read_queries(raw / "queries.jsonl"),
        _read_qrels(raw / "qrels" / "train.tsv"),
        _read_qrels(raw / "qrels" / "test.tsv"),
    )
    exact, normalized = compare_abstracts_to_beir_bodies(
        release.abstracts, load_beir_corpus_fields(raw / "corpus.jsonl")
    )
    if normalized:
        raise SystemExit("BEIR bodies do not match SciFact abstracts after whitespace collapsing")
    if mapping.train.qrel_equals_evidence == mapping.train.claim_count:
        raise SystemExit("refusing to select: BEIR qrels matched every evidence set")
    claims = release.train
    documents = load_documents(experiment.documents_path).documents
    depth = max(experiment.grid.doc_k)
    print(f"Retrieving top {depth} documents for {len(claims)} train claims", flush=True)
    retrieved, times = retrieve_top_documents(
        documents,
        [(claim.id, claim.text) for claim in claims],
        k=depth,
        k1=retrieval.k1,
        b=retrieval.b,
    )
    scorer = CrossEncoderNli(experiment.nli)
    jobs = build_sentence_jobs(claims, release.abstracts, retrieved, depth)
    print(f"Scoring {len(jobs)} train sentence/claim pairs", flush=True)
    started = perf_counter()
    rows = score_sentence_jobs(jobs, scorer)
    grid_seconds = perf_counter() - started
    grid_truncated = scorer.truncated_pairs
    scored = []
    for policy in iter_policies(experiment.grid):
        report = evaluate_predictions(
            claims,
            predictions_for_policy(claims, rows, policy),
            split="train",
            system="nli",
            model_name=experiment.nli.model_name,
            model_revision=str(scorer.metadata["nli_revision"]),
            pair_order="premise_sentence_hypothesis_claim",
            doc_k=policy.doc_k,
            sentence_k=policy.sentence_k,
            min_confidence=policy.min_confidence,
        )
        scored.append((policy, report))
        print(
            f"doc_k={policy.doc_k} sentence_k={policy.sentence_k} "
            f"min_confidence={policy.min_confidence} macro_f1={report.macro_f1:.4f} "
            f"sentence_f1={report.sentence_f1:.4f}",
            flush=True,
        )
    winner, winner_report = choose_policy(scored)
    if winner.doc_k == depth:
        winner_seconds = grid_seconds
        winner_pairs = len(jobs)
        winner_truncated = grid_truncated
    else:
        # The grid already has this policy's scores. Time the winning depth on its own.
        winner_jobs = build_sentence_jobs(claims, release.abstracts, retrieved, winner.doc_k)
        print(f"Timing winner doc_k={winner.doc_k} ({len(winner_jobs)} pairs)", flush=True)
        started = perf_counter()
        score_sentence_jobs(winner_jobs, scorer)
        winner_seconds = perf_counter() - started
        winner_truncated = scorer.truncated_pairs - grid_truncated
        winner_pairs = len(winner_jobs)
    label, counts = majority_verdict([claim.gold_verdict() for claim in claims])
    baseline = evaluate_predictions(
        claims,
        predict_baseline(claims, retrieved, release.abstracts, label),
        split="train",
        system="majority_baseline",
        model_name="majority_train_verdict",
        model_revision="none",
        pair_order="not_applicable",
        baseline_verdict=label,
        doc_k=1,
        sentence_k=1,
    )
    official = winner_report.model_copy(
        update={
            "retrieval_p95_ms": percentile(times, 0.95),
            "nli_pairs": winner_pairs,
            "nli_truncated_pairs": winner_truncated,
            "nli_seconds": winner_seconds,
            "claims_sha256": release.claims_train_sha256,
            "corpus_sha256": release.corpus_sha256,
            "environment": _environment(),
        }
    )
    payload = {
        "split": "train",
        "claim_count": len(claims),
        "selection_metric": "macro_f1",
        "tie_break": list(TIE_BREAK),
        "model_name": experiment.nli.model_name,
        "model_revision": scorer.metadata["nli_revision"],
        "max_length": experiment.nli.max_length,
        "pair_order": "premise_sentence_hypothesis_claim",
        "retriever": {
            "type": "bm25",
            "tokenizer": "stem",
            "unit": "document",
            "k1": retrieval.k1,
            "b": retrieval.b,
        },
        "mapping": {
            "train_qrel_equals_cited": mapping.train.qrel_equals_cited,
            "train_qrel_equals_evidence": mapping.train.qrel_equals_evidence,
            "dev_qrel_equals_cited": mapping.dev.qrel_equals_cited,
            "dev_qrel_equals_evidence": mapping.dev.qrel_equals_evidence,
            "exact_space_join_mismatches": exact,
            "whitespace_normalized_body_mismatches": normalized,
        },
        "baseline_verdict": label,
        "baseline_train_counts": counts,
        "baseline": {
            "accuracy": baseline.accuracy,
            "macro_f1": baseline.macro_f1,
            "sentence_f1": baseline.sentence_f1,
        },
        "grid_nli_pairs": len(jobs),
        "grid_nli_seconds": grid_seconds,
        "grid_truncated_pairs": grid_truncated,
        "candidates": [_row(policy, report) for policy, report in scored],
        "winner": _row(winner, official),
        "winner_nli_pairs": winner_pairs,
        "winner_nli_seconds": winner_seconds,
        "winner_truncated_pairs": winner_truncated,
        "retrieval_p95_ms": official.retrieval_p95_ms,
    }
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    selection_path = results / "verdict-train-selection.json"
    report_path = results / "verdict-train.json"
    selection_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(official.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"winner doc_k={winner.doc_k} sentence_k={winner.sentence_k} min_confidence={winner.min_confidence}"
    )
    print(f"baseline verdict={label} counts={counts}")
    print(f"Saved: {selection_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
