"""README regions stay aligned with the committed reports."""

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_readme_tables", ROOT / "scripts" / "render_readme_tables.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _documents(renderer: ModuleType) -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in renderer.DOCUMENTS}


def test_readme_regions_match_the_reports() -> None:
    renderer = _renderer()
    texts = _documents(renderer)
    assert renderer.render_documents(texts) == texts


def test_rendered_numbers_trace_to_report_fields() -> None:
    renderer = _renderer()
    texts = _documents(renderer)
    rendered = renderer.render_documents(texts)
    shown: dict[str, list[str]] = {}
    for citation in renderer.CITATIONS:
        value = renderer.lookup(renderer.load_report(citation.path), citation.expr)
        assert renderer.format_value(citation.fmt, value) == citation.shown
        shown.setdefault(citation.region, []).append(citation.shown)
    regions = [
        match for text in rendered.values() for match in renderer.REGION_PATTERN.finditer(text)
    ]
    assert {match.group("name") for match in regions} == set(renderer.RENDERERS)
    for match in regions:
        name = match.group("name")
        body = match.group("body")
        for token in shown[name]:
            assert token in body
        assert renderer.uncited_digits(body, shown[name]) == ""


def test_readme_prose_outside_regions_has_no_long_decimals() -> None:
    renderer = _renderer()
    for text in _documents(renderer).values():
        stripped = renderer.REGION_PATTERN.sub("", text)
        stripped = re.sub(r"```.*?```", "", stripped, flags=re.DOTALL)
        stripped = re.sub(r"https?://\S+", "", stripped)
        assert re.search(r"\d+\.\d{4,}", stripped) is None


def test_held_out_scifact_reports_share_the_test_fingerprint() -> None:
    baseline = _load("benchmarks/scifact/bm25.json")
    for path in (
        "benchmarks/scifact/bm25-document.json",
        "benchmarks/scifact/bm25-document-k1.5-b0.75.json",
        "benchmarks/scifact/bm25-selected.json",
        "benchmarks/scifact/hybrid-selected.json",
        "benchmarks/scifact/rerank-selected.json",
    ):
        report = _load(path)
        assert report["benchmark_sha256"] == baseline["benchmark_sha256"]
        assert report["corpus_sha256"] == baseline["corpus_sha256"]
        assert report["question_count"] == 300
