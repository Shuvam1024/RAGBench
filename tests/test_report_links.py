"""Committed comparisons point at their reports, and doc links resolve."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Comparison file -> (baseline report, candidate report).
COMPARISON_REPORTS = {
    "benchmarks/scifact/hybrid-vs-rerank.json": (
        "benchmarks/scifact/hybrid-selected.json",
        "benchmarks/scifact/rerank-selected.json",
    ),
    "benchmarks/scifact/document-vs-bm25-selected.json": (
        "benchmarks/scifact/bm25-document.json",
        "benchmarks/scifact/bm25-selected.json",
    ),
    "benchmarks/scifact/document-vs-hybrid-selected.json": (
        "benchmarks/scifact/bm25-document.json",
        "benchmarks/scifact/hybrid-selected.json",
    ),
    "benchmarks/scifact/bm25-selected-vs-hybrid.json": (
        "benchmarks/scifact/bm25-selected.json",
        "benchmarks/scifact/hybrid-selected.json",
    ),
    "benchmarks/scifact/document-vs-bm25-k15.json": (
        "benchmarks/scifact/bm25-document.json",
        "benchmarks/scifact/bm25-document-k1.5-b0.75.json",
    ),
    "benchmarks/scifact/bm25-k15-vs-selected.json": (
        "benchmarks/scifact/bm25-document-k1.5-b0.75.json",
        "benchmarks/scifact/bm25-selected.json",
    ),
    "benchmarks/scifact/bm25-vs-hybrid-w075.json": (
        "benchmarks/scifact/bm25.json",
        "benchmarks/scifact/hybrid-w075.json",
    ),
    "benchmarks/nfcorpus/document-vs-bm25-selected.json": (
        "benchmarks/nfcorpus/bm25-document.json",
        "benchmarks/nfcorpus/bm25-selected.json",
    ),
    "benchmarks/nfcorpus/document-vs-hybrid-selected.json": (
        "benchmarks/nfcorpus/bm25-document.json",
        "benchmarks/nfcorpus/hybrid-selected.json",
    ),
    "benchmarks/nfcorpus/bm25-selected-vs-hybrid.json": (
        "benchmarks/nfcorpus/bm25-selected.json",
        "benchmarks/nfcorpus/hybrid-selected.json",
    ),
    "benchmarks/nfcorpus/document-vs-bm25-k15.json": (
        "benchmarks/nfcorpus/bm25-document.json",
        "benchmarks/nfcorpus/bm25-document-k1.5-b0.75.json",
    ),
    "benchmarks/nfcorpus/bm25-k15-vs-selected.json": (
        "benchmarks/nfcorpus/bm25-document-k1.5-b0.75.json",
        "benchmarks/nfcorpus/bm25-selected.json",
    ),
}

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DOCUMENTS = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))


def _load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _slug(title: str) -> str:
    text = title.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def _heading_slugs(text: str) -> set[str]:
    slugs: set[str] = set()
    counts: dict[str, int] = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line.startswith("#"):
            continue
        slug = _slug(line.lstrip("#"))
        seen = counts.get(slug, 0)
        counts[slug] = seen + 1
        slugs.add(slug if seen == 0 else f"{slug}-{seen}")
    return slugs


def _visible_lines(text: str) -> list[str]:
    lines: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def test_comparison_run_ids_match_their_reports() -> None:
    found = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "benchmarks").rglob("*.json")
        if "baseline_run_id" in json.loads(path.read_text(encoding="utf-8"))
    )
    assert found == sorted(COMPARISON_REPORTS)
    for comparison_path, (baseline_path, candidate_path) in COMPARISON_REPORTS.items():
        comparison = _load(comparison_path)
        assert comparison["baseline_run_id"] == _load(baseline_path)["run_id"]
        assert comparison["candidate_run_id"] == _load(candidate_path)["run_id"]


def test_relative_doc_links_resolve() -> None:
    missing: list[str] = []
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for line in _visible_lines(text):
            for match in LINK.finditer(line):
                target = match.group(1).split()[0]
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                file_part, _, fragment = target.partition("#")
                destination = path.parent / file_part if file_part else path
                destination = destination.resolve()
                label = f"{path.relative_to(ROOT)} -> {target}"
                if not destination.is_file():
                    missing.append(label)
                    continue
                if fragment and destination.suffix == ".md":
                    slugs = _heading_slugs(destination.read_text(encoding="utf-8"))
                    if fragment not in slugs:
                        missing.append(f"{label} (no heading)")
    assert missing == []
