"""Dev-only oracle retrieval and an error breakdown for the frozen NLI policy.

The policy is the one frozen on the train claims. This script does not choose
a knob, and it does not score the public test split. The oracle shows every
gold evidence document to that same sentence policy.
"""

import json
from pathlib import Path
from time import perf_counter

from ragstat.datasets.scifact_claims import prepare_scifact_release
from ragstat.evaluation.nli import CrossEncoderNli
from ragstat.evaluation.verdict import (
    VerdictDevAnalysis,
    VerdictPolicy,
    breakdown_errors,
    build_oracle_jobs,
    evaluate_predictions,
    load_verdict_experiment,
    metric_snapshot,
    predictions_for_policy,
    retrieve_top_documents,
    score_sentence_jobs,
)
from ragstat.ingestion.loader import load_documents

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scifact-verdict-dev.yaml"
FROZEN = ROOT / "benchmarks" / "scifact" / "verdict-dev.json"
BASELINE = ROOT / "benchmarks" / "scifact" / "verdict-baseline-dev.json"
ORACLE = ROOT / "benchmarks" / "scifact" / "verdict-oracle-dev.json"
ANALYSIS = ROOT / "benchmarks" / "scifact" / "verdict-dev-analysis.json"
NOTE = (
    "Dev is held out. The oracle is not a selected system. It gives the frozen "
    "sentence policy every gold evidence document and leaves sentence_k and "
    "min_confidence unchanged. NEI claims have no gold document, so the oracle "
    "predicts NEI for them."
)


def main() -> None:
    experiment = load_verdict_experiment(CONFIG)
    if experiment.split != "dev" or experiment.policy is None:
        raise SystemExit("refusing to analyze: the dev file must name a frozen policy")
    policy = experiment.policy
    release = prepare_scifact_release(experiment.claims_cache)
    claims = release.dev
    documents = load_documents(experiment.documents_path).documents
    print(f"Retrieving top {policy.doc_k} for {len(claims)} dev claims", flush=True)
    retrieved, _times = retrieve_top_documents(
        documents,
        [(claim.id, claim.text) for claim in claims],
        k=policy.doc_k,
        k1=experiment.retrieval.k1,
        b=experiment.retrieval.b,
    )
    from ragstat.evaluation.verdict import VerdictEvaluation

    frozen = VerdictEvaluation.model_validate_json(FROZEN.read_text(encoding="utf-8"))
    baseline = VerdictEvaluation.model_validate_json(BASELINE.read_text(encoding="utf-8"))
    if [claim.id for claim in frozen.claims] != [claim.id for claim in claims]:
        raise SystemExit("committed dev report does not match the dev claims")
    errors = breakdown_errors(frozen.claims, retrieved, doc_k=policy.doc_k)
    scorer = CrossEncoderNli(experiment.nli)
    jobs = build_oracle_jobs(claims, release.abstracts)
    print(f"Scoring {len(jobs)} oracle sentence/claim pairs", flush=True)
    started = perf_counter()
    rows = score_sentence_jobs(jobs, scorer)
    elapsed = perf_counter() - started
    revision = str(scorer.metadata["nli_revision"])
    if revision != experiment.nli.revision:
        raise SystemExit(f"resolved NLI revision {revision} does not match the pin")
    oracle = evaluate_predictions(
        claims,
        predictions_for_policy(claims, rows, VerdictPolicy.model_validate(policy.model_dump())),
        split="dev",
        system="oracle_retrieval",
        model_name=experiment.nli.model_name,
        model_revision=revision,
        pair_order="premise_sentence_hypothesis_claim",
        doc_k=policy.doc_k,
        sentence_k=policy.sentence_k,
        min_confidence=policy.min_confidence,
    ).model_copy(
        update={
            "nli_pairs": len(jobs),
            "nli_truncated_pairs": scorer.truncated_pairs,
            "nli_seconds": elapsed,
            "claims_sha256": release.claims_dev_sha256,
            "corpus_sha256": release.corpus_sha256,
        }
    )
    analysis = VerdictDevAnalysis(
        doc_k=policy.doc_k,
        frozen_nli=metric_snapshot(frozen),
        majority_baseline=metric_snapshot(baseline),
        oracle_retrieval=metric_snapshot(oracle),
        errors=errors,
        note=NOTE,
    )
    ORACLE.write_text(oracle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ANALYSIS.write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis.model_dump(), indent=2))
    print(f"Saved: {ORACLE}")
    print(f"Saved: {ANALYSIS}")


if __name__ == "__main__":
    main()
