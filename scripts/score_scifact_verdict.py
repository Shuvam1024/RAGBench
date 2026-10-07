"""Score the frozen SciFact verdict policy once on the dev claims.

The policy and the baseline verdict come from the train selection file. This
script does not choose them again. The public test file stays unscored.
"""

import json
import math
from pathlib import Path
from time import perf_counter

from ragbench.datasets.beir import _read_qrels, _read_queries
from ragbench.datasets.scifact_claims import (
    compare_abstracts_to_beir_bodies,
    load_beir_corpus_fields,
    prepare_scifact_release,
    require_scifact_beir_mapping,
)
from ragbench.evaluation.nli import CrossEncoderNli
from ragbench.evaluation.runner import pytorch_threads
from ragbench.evaluation.timing import percentile
from ragbench.evaluation.verdict import (
    VerdictPolicy,
    build_sentence_jobs,
    compare_verdicts,
    evaluate_predictions,
    load_verdict_experiment,
    majority_verdict,
    predict_baseline,
    predictions_for_policy,
    retrieve_top_documents,
    score_sentence_jobs,
)
from ragbench.ingestion.loader import load_documents

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scifact-verdict-dev.yaml"
SELECTION = ROOT / "benchmarks" / "scifact" / "verdict-train-selection.json"


def _environment() -> dict[str, str]:
    import os
    import platform

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
        "pytorch_threads": pytorch_threads(),
    }


def main() -> None:
    if not SELECTION.is_file():
        raise SystemExit(f"Missing train selection file {SELECTION}")
    experiment = load_verdict_experiment(CONFIG)
    if (
        experiment.split != "dev"
        or experiment.policy is None
        or experiment.baseline_verdict is None
    ):
        raise SystemExit("refusing to score: the dev file must name a frozen policy")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    winner = selection["winner"]
    policy = experiment.policy
    if (
        policy.doc_k != winner["doc_k"]
        or policy.sentence_k != winner["sentence_k"]
        or not math.isclose(policy.min_confidence, float(winner["min_confidence"]), abs_tol=1e-12)
        or experiment.baseline_verdict != selection["baseline_verdict"]
        or experiment.nli.model_name != selection["model_name"]
        or experiment.nli.revision != selection["model_revision"]
    ):
        raise SystemExit("refusing to score: dev config does not match the train winner")
    release = prepare_scifact_release(experiment.claims_cache)
    raw = experiment.beir_raw
    require_scifact_beir_mapping(
        release.train,
        release.dev,
        release.test,
        _read_queries(raw / "queries.jsonl"),
        _read_qrels(raw / "qrels" / "train.tsv"),
        _read_qrels(raw / "qrels" / "test.tsv"),
    )
    _exact, normalized = compare_abstracts_to_beir_bodies(
        release.abstracts, load_beir_corpus_fields(raw / "corpus.jsonl")
    )
    if normalized:
        raise SystemExit("BEIR bodies do not match SciFact abstracts after whitespace collapsing")
    label, counts = majority_verdict([claim.gold_verdict() for claim in release.train])
    if label != experiment.baseline_verdict or counts != selection["baseline_train_counts"]:
        raise SystemExit("refusing to score: train majority does not match the frozen baseline")
    claims = release.dev
    documents = load_documents(experiment.documents_path).documents
    print(f"Retrieving top {policy.doc_k} documents for {len(claims)} dev claims", flush=True)
    retrieved, times = retrieve_top_documents(
        documents,
        [(claim.id, claim.text) for claim in claims],
        k=policy.doc_k,
        k1=experiment.retrieval.k1,
        b=experiment.retrieval.b,
    )
    scorer = CrossEncoderNli(experiment.nli)
    jobs = build_sentence_jobs(claims, release.abstracts, retrieved, policy.doc_k)
    print(f"Scoring {len(jobs)} dev sentence/claim pairs", flush=True)
    started = perf_counter()
    rows = score_sentence_jobs(jobs, scorer)
    nli_seconds = perf_counter() - started
    revision = str(scorer.metadata["nli_revision"])
    if revision != experiment.nli.revision:
        raise SystemExit(f"resolved NLI revision {revision} does not match the pin")
    candidate = evaluate_predictions(
        claims,
        predictions_for_policy(claims, rows, VerdictPolicy.model_validate(policy.model_dump())),
        split="dev",
        system="nli",
        model_name=experiment.nli.model_name,
        model_revision=revision,
        pair_order="premise_sentence_hypothesis_claim",
        doc_k=policy.doc_k,
        sentence_k=policy.sentence_k,
        min_confidence=policy.min_confidence,
    ).model_copy(
        update={
            "retrieval_p95_ms": percentile(times, 0.95),
            "nli_pairs": len(jobs),
            "nli_truncated_pairs": scorer.truncated_pairs,
            "nli_seconds": nli_seconds,
            "claims_sha256": release.claims_dev_sha256,
            "corpus_sha256": release.corpus_sha256,
            "environment": _environment(),
        }
    )
    baseline = evaluate_predictions(
        claims,
        predict_baseline(claims, retrieved, release.abstracts, label),
        split="dev",
        system="majority_baseline",
        model_name="majority_train_verdict",
        model_revision="none",
        pair_order="not_applicable",
        baseline_verdict=label,
        doc_k=1,
        sentence_k=1,
    ).model_copy(
        update={
            "retrieval_p95_ms": candidate.retrieval_p95_ms,
            "claims_sha256": release.claims_dev_sha256,
            "corpus_sha256": release.corpus_sha256,
            "environment": candidate.environment,
        }
    )
    comparison = compare_verdicts(baseline, candidate)
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name, report in (
        ("verdict-dev.json", candidate),
        ("verdict-baseline-dev.json", baseline),
    ):
        path = results / name
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {path}")
    comparison_path = results / "verdict-baseline-vs-nli.json"
    comparison_path.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {comparison_path}")
    print(
        f"dev accuracy={candidate.accuracy:.4f} macro_f1={candidate.macro_f1:.4f} "
        f"sentence_f1={candidate.sentence_f1:.4f}"
    )


if __name__ == "__main__":
    main()
