"""README tables stay aligned with the committed held-out reports."""

import json
from pathlib import Path

README = Path("README.md").read_text(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

REPORTS = (
    "benchmarks/scifact/bm25.json",
    "benchmarks/scifact/bm25-document.json",
    "benchmarks/scifact/bm25-selected.json",
    "benchmarks/scifact/hybrid-selected.json",
    "benchmarks/scifact/bm25-stem-chunk120-exploratory.json",
    "benchmarks/nfcorpus/bm25-whitespace.json",
    "benchmarks/nfcorpus/bm25-document.json",
    "benchmarks/nfcorpus/bm25-selected.json",
    "benchmarks/nfcorpus/hybrid-selected.json",
)

COMPARISONS = (
    "benchmarks/scifact/document-vs-bm25-selected.json",
    "benchmarks/scifact/document-vs-hybrid-selected.json",
    "benchmarks/scifact/bm25-selected-vs-hybrid.json",
    "benchmarks/nfcorpus/document-vs-bm25-selected.json",
    "benchmarks/nfcorpus/document-vs-hybrid-selected.json",
)


def _load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_readme_quotes_report_metrics_at_four_decimals() -> None:
    for path in REPORTS:
        report = _load(path)
        ndcg = report["ndcg_at_k"]
        recall = report["recall_at_k"]
        assert isinstance(ndcg, dict) and isinstance(recall, dict)
        assert f"{ndcg['10']:.4f}" in README
        assert f"{recall['10']:.4f}" in README
        assert f"{report['mean_average_precision']:.4f}" in README
        candidate_recall = report.get("candidate_recall_at_100")
        if isinstance(candidate_recall, float):
            assert f"{candidate_recall:.4f}" in README


def test_readme_quotes_paired_deltas() -> None:
    for path in COMPARISONS:
        comparison = _load(path)
        checks = comparison["checks"]
        assert isinstance(checks, list)
        for check in checks:
            assert isinstance(check, dict)
            delta = f"{check['mean_delta']:+.4f}"
            interval = f"[{check['ci_low']:+.4f}, {check['ci_high']:+.4f}]"
            assert delta in README
            assert interval in README


def test_held_out_scifact_reports_share_the_test_fingerprint() -> None:
    baseline = _load("benchmarks/scifact/bm25.json")
    for path in (
        "benchmarks/scifact/bm25-document.json",
        "benchmarks/scifact/bm25-selected.json",
        "benchmarks/scifact/hybrid-selected.json",
    ):
        report = _load(path)
        assert report["benchmark_sha256"] == baseline["benchmark_sha256"]
        assert report["corpus_sha256"] == baseline["corpus_sha256"]
        assert report["question_count"] == 300
