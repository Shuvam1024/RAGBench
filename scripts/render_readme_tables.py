"""Rewrite marked README regions from the committed JSON reports.

Metric figures use 3 decimal places. Millisecond latencies are whole
milliseconds. Second latencies use 1 decimal place. Sign-flip p-values stay
at 4 decimals because the permutation floor is 1/10,000. The JSON files are
not modified.

Each displayed number is produced from one report field (or a product or
difference of two fields). ``<!-- tables:begin name -->`` and
``<!-- tables:end name -->`` mark the regions this script owns.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "docs" / "results.md"
DOCUMENTS = (README, RESULTS)
REGION_PATTERN = re.compile(
    r"<!-- tables:begin (?P<name>[a-z0-9-]+) -->\n(?P<body>.*?)\n<!-- tables:end (?P=name) -->",
    re.DOTALL,
)

LABEL_FRAGMENTS = (
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "Recall@10",
    "Recall@100",
    "nDCG@10",
    "P@10",
    "Candidate Recall@100",
    "p95",
    "BM25",
    "F1",
)

RANKING_METRICS = ("ndcg@10", "recall@10", "recall@100", "map", "mrr")
METRIC_LABELS = {
    "ndcg@10": "nDCG@10",
    "recall@10": "Recall@10",
    "recall@100": "Recall@100",
    "map": "MAP",
    "mrr": "MRR",
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "sentence_precision": "Sentence precision",
    "sentence_recall": "Sentence recall",
    "sentence_f1": "Sentence F1",
}
VERDICT_METRICS = (
    "accuracy",
    "macro_f1",
    "sentence_precision",
    "sentence_recall",
    "sentence_f1",
)


@dataclass(frozen=True)
class Citation:
    region: str
    path: str
    expr: str
    fmt: str
    shown: str


CITATIONS: list[Citation] = []
CURRENT_REGION = ""
_CACHE: dict[str, dict[str, object]] = {}


def load_report(path: str) -> dict[str, object]:
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    file = ROOT / path
    if path.endswith((".yaml", ".yml")):
        loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must be a YAML mapping")
        parsed = loaded
    else:
        loaded = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must be a JSON object")
        parsed = loaded
    _CACHE[path] = parsed
    return parsed


def _coerce(raw: str) -> object:
    if raw in {"true", "false"}:
        return raw == "true"
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer() and "." not in raw:
        return int(number)
    return number


def _step(current: object, part: str) -> object:
    if "[" not in part:
        if not isinstance(current, dict):
            raise KeyError(part)
        return current[part]
    name, rest = part.split("[", 1)
    spec = rest[:-1]
    if not isinstance(current, dict):
        raise KeyError(part)
    parent: object = current[name] if name else current
    if not isinstance(parent, list):
        raise KeyError(part)
    if "=" not in spec:
        return parent[int(spec)]
    key, raw = spec.split("=", 1)
    want = _coerce(raw)
    for item in parent:
        if isinstance(item, dict) and (item.get(key) == want or str(item.get(key)) == raw):
            return item
    raise KeyError(part)


def _parts(expr: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in expr:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "." and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Expected a number, got {value!r}")
    return float(value)


def _count(data: object, expr: str) -> int:
    """Count list rows matching ``name[key=value][key!=value]`` filters."""
    head, _, _rest = expr.partition("[")
    rows = lookup(data, head)
    if not isinstance(rows, list):
        raise ValueError(f"{expr} does not name a list")
    matched = 0
    specs = re.findall(r"\[([^\]]+)\]", expr[len(head) :])
    parsed: list[tuple[str, object, str, bool]] = []
    for spec in specs:
        if "!=" in spec:
            key, raw = spec.split("!=", 1)
            negate = True
        elif "=" in spec:
            key, raw = spec.split("=", 1)
            negate = False
        else:
            raise ValueError(f"Cannot filter {expr}")
        parsed.append((key, _coerce(raw), raw, negate))
    for item in rows:
        if not isinstance(item, dict):
            continue
        keep = True
        for key, want, raw, negate in parsed:
            equal = item.get(key) == want or str(item.get(key)) == raw
            if equal == negate:
                keep = False
                break
        if keep:
            matched += 1
    return matched


def lookup(data: object, expr: str) -> object:
    """Resolve a dotted field, with one product or difference when needed."""
    if expr.startswith("count:"):
        return _count(data, expr.removeprefix("count:"))
    if " * " in expr:
        left, right = expr.split(" * ", 1)
        return _number(lookup(data, left)) * _number(lookup(data, right))
    if " - " in expr:
        left, right = expr.split(" - ", 1)
        return _number(lookup(data, left)) - _number(lookup(data, right))
    current = data
    for part in _parts(expr):
        current = _step(current, part)
    return current


def format_value(fmt: str, value: object) -> str:
    if fmt == "pvalue":
        if value is None:
            return "none"
        return f"{_number(value):.4f}"
    if fmt == "metric":
        return f"{_number(value):.3f}"
    if fmt == "signed":
        # A value inside ±0.0005 keeps its sign, so -0.00031 renders as -0.000.
        return f"{_number(value):+.3f}"
    if fmt == "metric4":
        return f"{_number(value):.4f}"
    if fmt == "ms":
        return f"{round(_number(value)):,}"
    if fmt == "seconds":
        return f"{_number(value):.1f}"
    if fmt == "count":
        return f"{int(_number(value)):,}"
    if fmt == "weight":
        return f"{_number(value):.2f}"
    if fmt == "threshold":
        return f"{_number(value):.1f}"
    if fmt == "percent":
        return f"{_number(value) * 100:.0f}"
    if fmt == "usd":
        return f"{_number(value):.6f}"
    if fmt == "ms_to_seconds":
        return f"{_number(value) / 1000:.1f}"
    if fmt == "prefix16":
        return str(value)[:16]
    raise ValueError(f"Unknown format {fmt}")


def cite(path: str, expr: str, fmt: str) -> str:
    shown = format_value(fmt, lookup(load_report(path), expr))
    CITATIONS.append(Citation(CURRENT_REGION, path, expr, fmt, shown))
    return shown


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def ranking_cells(path: str) -> list[str]:
    return [
        cite(path, "recall_at_k.10", "metric"),
        cite(path, "recall_at_k.100", "metric"),
        cite(path, "ndcg_at_k.10", "metric"),
        cite(path, "precision_at_k.10", "metric"),
        cite(path, "mean_average_precision", "metric"),
        cite(path, "mrr", "metric"),
    ]


def _checks(path: str) -> list[dict[str, object]]:
    rows = lookup(load_report(path), "checks")
    if not isinstance(rows, list):
        raise ValueError(f"{path} checks must be a list")
    order = {name: index for index, name in enumerate(RANKING_METRICS + VERDICT_METRICS)}
    typed = [row for row in rows if isinstance(row, dict)]
    return sorted(typed, key=lambda row: order.get(str(row["metric"]), 99))


def comparison_table(path: str) -> str:
    rows: list[list[str]] = []
    for check in _checks(path):
        metric = str(check["metric"])
        rows.append(
            [
                METRIC_LABELS[metric],
                cite(path, f"checks[metric={metric}].mean_delta", "signed"),
                (
                    "["
                    + cite(path, f"checks[metric={metric}].ci_low", "signed")
                    + ", "
                    + cite(path, f"checks[metric={metric}].ci_high", "signed")
                    + "]"
                ),
                cite(path, f"checks[metric={metric}].permutation_p_value", "pvalue"),
            ]
        )
    return table(["Metric", "Mean delta", "95% CI", "Sign-flip p"], rows)


def interval_notes(path: str) -> str:
    above: list[str] = []
    below: list[str] = []
    includes: list[str] = []
    rounded: list[str] = []
    for check in _checks(path):
        metric = str(check["metric"])
        label = METRIC_LABELS[metric]
        low = _number(check["ci_low"])
        high = _number(check["ci_high"])
        if low > 0:
            above.append(label)
            if float(f"{low:.3f}") == 0.0:
                shown = cite(path, f"checks[metric={metric}].ci_low", "signed")
                rounded.append(f"{label} is above zero, but its lower bound rounds to {shown}.")
        elif high < 0:
            below.append(label)
            if float(f"{high:.3f}") == 0.0:
                shown = cite(path, f"checks[metric={metric}].ci_high", "signed")
                rounded.append(f"{label} is below zero, but its upper bound rounds to {shown}.")
        else:
            includes.append(label)
    sentences: list[str] = []
    if above:
        sentences.append("Above zero: " + ", ".join(above) + ".")
    if below:
        sentences.append("Below zero: " + ", ".join(below) + ".")
    if includes:
        sentences.append("Intervals that include zero: " + ", ".join(includes) + ".")
    sentences.extend(rounded)
    return " ".join(sentences)


def _strictly_positive(path: str) -> bool:
    checks = _checks(path)
    names = {str(check["metric"]) for check in checks}
    if names != set(RANKING_METRICS):
        raise ValueError(f"{path} does not contain the five ranking metrics")
    return all(_number(check["ci_low"]) > 0 for check in checks)


def render_at_a_glance() -> str:
    hybrid = "benchmarks/scifact/hybrid-selected.json"
    document = "benchmarks/scifact/bm25-document.json"
    sci_hybrid = "benchmarks/scifact/document-vs-hybrid-selected.json"
    nf_hybrid = "benchmarks/nfcorpus/document-vs-hybrid-selected.json"
    nf_chunk = "benchmarks/nfcorpus/document-vs-bm25-selected.json"
    ablation = "benchmarks/scifact/bm25-document-k1.5-b0.75.json"
    sci_k15 = "benchmarks/scifact/document-vs-bm25-k15.json"
    sci_chunk_vs_k15 = "benchmarks/scifact/bm25-k15-vs-selected.json"
    nf_k15 = "benchmarks/nfcorpus/document-vs-bm25-k15.json"
    rerank_cmp = "benchmarks/scifact/hybrid-vs-rerank.json"
    rerank = "benchmarks/scifact/rerank-selected.json"
    sci_stats = "benchmarks/scifact/corpus_stats.json"
    nf_stats = "benchmarks/nfcorpus/corpus_stats.json"
    train = "benchmarks/scifact/bm25-train-selection.json"
    selected_cfg = "configs/scifact-bm25-selected.yaml"
    document_cfg = "configs/scifact-bm25-document.yaml"
    nli = "benchmarks/scifact/verdict-dev.json"
    majority = "benchmarks/scifact/verdict-baseline-dev.json"
    paired = "benchmarks/scifact/verdict-baseline-vs-nli.json"
    recipe = "configs/paired-uncertainty.yaml"
    if lookup(load_report(recipe), "statistics.confidence") != lookup(
        load_report(paired), "confidence"
    ):
        raise ValueError(f"{recipe} confidence does not match {paired}")
    if not _strictly_positive(nf_hybrid):
        raise ValueError(f"{nf_hybrid} is not above zero on every ranking metric")
    if _strictly_positive(nf_chunk):
        raise ValueError(f"{nf_chunk} is above zero on every ranking metric")
    accuracy_low = _number(lookup(load_report(paired), "checks[metric=accuracy].ci_low"))
    accuracy_high = _number(lookup(load_report(paired), "checks[metric=accuracy].ci_high"))
    if not accuracy_low < 0 < accuracy_high:
        raise ValueError("accuracy interval does not include zero")
    nf_k15_low = _number(lookup(load_report(nf_k15), "checks[metric=ndcg@10].ci_low"))
    nf_k15_high = _number(lookup(load_report(nf_k15), "checks[metric=ndcg@10].ci_high"))
    if not nf_k15_low < 0 < nf_k15_high:
        raise ValueError("NFCorpus parameter ablation interval does not include zero")
    sci = "[SciFact test split](docs/results.md#held-out-scifact-test)"
    nf = "[NFCorpus](docs/results.md#nfcorpus-confirmation)"
    dev = "[SciFact dev](docs/results.md#held-out-scifact-dev)"
    level = cite(paired, "confidence", "percent")
    return "\n".join(
        [
            (
                "- Frozen hybrid (BM25 + MiniLM, chosen on train, test scored once) versus "
                f"document-level BM25 on the {sci}: nDCG@10 "
                f"{cite(hybrid, 'ndcg_at_k.10', 'metric')} vs "
                f"{cite(document, 'ndcg_at_k.10', 'metric')}, delta "
                f"{cite(sci_hybrid, 'checks[metric=ndcg@10].mean_delta', 'signed')}, "
                f"{level}% CI ["
                f"{cite(sci_hybrid, 'checks[metric=ndcg@10].ci_low', 'signed')}, "
                f"{cite(sci_hybrid, 'checks[metric=ndcg@10].ci_high', 'signed')}]."
            ),
            (
                "- The same frozen settings on "
                f"{nf}: hybrid nDCG@10 delta "
                f"{cite(nf_hybrid, 'checks[metric=ndcg@10].mean_delta', 'signed')} "
                "over document-level BM25, with the CI above zero on all five metrics."
            ),
            (
                "- The selected chunked index is nearly one chunk per document: only "
                f"{cite(sci_stats, 'word_length.longer_than[words=480].documents', 'count')} of "
                f"{cite(sci_stats, 'document_count', 'count')} SciFact documents ("
                f"{cite(nf_stats, 'word_length.longer_than[words=480].documents', 'count')} of "
                f"{cite(nf_stats, 'document_count', 'count')} on NFCorpus) exceed the "
                f"{cite(train, 'winner.chunk_size', 'count')}-word window. A post-hoc ablation, "
                "run after the test numbers were reported and not used to select a setting, "
                "scores document-level BM25 with parameters "
                f"{cite(selected_cfg, 'retrieval.k1', 'weight')}/"
                f"{cite(selected_cfg, 'retrieval.b', 'weight')}. On the {sci} its nDCG@10 is "
                f"{cite(ablation, 'ndcg_at_k.10', 'metric4')}, the selected chunked index is "
                f"{cite('benchmarks/scifact/bm25-selected.json', 'ndcg_at_k.10', 'metric4')} "
                "(they differ on "
                f"{cite(sci_chunk_vs_k15, 'count:question_changes[metric=ndcg@10][direction!=unchanged]', 'count')} "
                f"of {cite(ablation, 'question_count', 'count')} queries), and document-level "
                f"BM25 at {cite(document_cfg, 'retrieval.k1', 'weight')}/"
                f"{cite(document_cfg, 'retrieval.b', 'weight')} is "
                f"{cite(document, 'ndcg_at_k.10', 'metric')}. The gap of "
                f"{cite(sci_k15, 'checks[metric=ndcg@10].mean_delta', 'signed')} is that "
                f"parameter change, {level}% CI ["
                f"{cite(sci_k15, 'checks[metric=ndcg@10].ci_low', 'signed')}, "
                f"{cite(sci_k15, 'checks[metric=ndcg@10].ci_high', 'signed')}]. "
                "Chunking adds about nothing on this split. On train the frozen pair tie at "
                f"{cite(train, 'winner.metrics.ndcg@10', 'metric4')} vs "
                f"{cite(train, 'document_level_reference.metrics.ndcg@10', 'metric4')}. "
                "The same ablation on "
                f"{nf} changes document-level nDCG@10 by "
                f"{cite(nf_k15, 'checks[metric=ndcg@10].mean_delta', 'signed')}, "
                "and that interval includes zero."
            ),
            (
                "- Cross-encoder rerank, not fine-tuned, is a negative result on the "
                f"{sci}: nDCG@10 "
                f"{cite(rerank_cmp, 'checks[metric=ndcg@10].mean_delta', 'signed')}, "
                f"{cite(paired, 'confidence', 'percent')}% CI ["
                f"{cite(rerank_cmp, 'checks[metric=ndcg@10].ci_low', 'signed')}, "
                f"{cite(rerank_cmp, 'checks[metric=ndcg@10].ci_high', 'signed')}], "
                f"rerank p95 {cite(rerank, 'timings.rerank_p95_ms', 'ms_to_seconds')} s."
            ),
            (
                f"- {dev} claim verdicts: frozen NLI micro sentence F1 "
                f"{cite(nli, 'micro_sentence_f1', 'metric')} vs "
                f"{cite(majority, 'micro_sentence_f1', 'metric')} for the majority baseline. "
                "Evidence-only sentence F1, the mean over claims with gold evidence, is "
                f"{cite(nli, 'evidence_only_sentence_f1', 'metric')} vs "
                f"{cite(majority, 'evidence_only_sentence_f1', 'metric')}. "
                "The per-claim mean is "
                f"{cite(nli, 'sentence_f1', 'metric')} vs "
                f"{cite(majority, 'sentence_f1', 'metric')} because a claim with no gold "
                "evidence and no predicted evidence scores as a perfect match. Macro-F1 is "
                f"{cite(nli, 'macro_f1', 'metric')} vs "
                f"{cite(majority, 'macro_f1', 'metric')}. "
                "The accuracy interval includes zero."
            ),
        ]
    )


def render_support_fixture() -> str:
    path = "benchmarks/baseline.json"
    return "\n".join(
        [
            "```text",
            f"Recall@1   {cite(path, 'recall_at_k.1', 'metric')}",
            f"Recall@3   {cite(path, 'recall_at_k.3', 'metric')}",
            f"Recall@5   {cite(path, 'recall_at_k.5', 'metric')}",
            f"MRR        {cite(path, 'mrr', 'metric')} (full document ranking)",
            "```",
        ]
    )


def render_lexical_train() -> str:
    path = "benchmarks/scifact/bm25-train-selection.json"
    headers = ["Candidate", "Train nDCG@10", "Recall@10", "Recall@100", "MAP", "MRR"]

    def row(label: str, prefix: str) -> list[str]:
        return [
            label,
            cite(path, f"{prefix}.ndcg@10", "metric"),
            cite(path, f"{prefix}.recall@10", "metric"),
            cite(path, f"{prefix}.recall@100", "metric"),
            cite(path, f"{prefix}.map", "metric"),
            cite(path, f"{prefix}.mrr", "metric"),
        ]

    return table(
        headers,
        [
            row(
                "Selected chunked (stem, " + cite(path, "winner.chunk_size", "count") + ")",
                "winner.metrics",
            ),
            row("Document-level reference", "document_level_reference.metrics"),
        ],
    )


def render_hybrid_weights() -> str:
    path = "benchmarks/scifact/hybrid-weight-train.json"
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = [
        [
            cite(path, f"rows[dense_weight={weight}].dense_weight", "weight"),
            cite(path, f"rows[dense_weight={weight}].ndcg@10", "metric"),
        ]
        for weight in weights
    ]
    grid = table(["dense_weight", "Train nDCG@10"], rows)
    frozen = cite(path, "dense_weight", "weight")
    return f"{grid}\n\nThe frozen weight is {frozen}."


def render_rerank_train() -> str:
    path = "benchmarks/scifact/rerank-train-selection.json"
    hybrid = "benchmarks/scifact/hybrid-weight-train.json"
    report = "benchmarks/scifact/rerank-train.json"
    rows = []
    for depth in (20, 50, 100):
        prefix = f"candidates[candidate_k={depth}].metrics"
        rows.append(
            [
                cite(path, f"candidates[candidate_k={depth}].candidate_k", "count"),
                cite(path, f"{prefix}.ndcg@10", "metric"),
                cite(path, f"{prefix}.recall@10", "metric"),
                cite(path, f"{prefix}.recall@100", "metric"),
                cite(path, f"{prefix}.candidate_recall_at_100", "metric"),
                cite(path, f"{prefix}.map", "metric"),
                cite(path, f"{prefix}.mrr", "metric"),
            ]
        )
    grid = table(
        [
            "candidate_k",
            "Train nDCG@10",
            "Recall@10",
            "Recall@100",
            "Candidate Recall@100",
            "MAP",
            "MRR",
        ],
        rows,
    )
    prose = "\n".join(
        [
            "",
            (
                "The frozen hybrid's train nDCG@10 is "
                f"{cite(hybrid, 'rows[dense_weight=0.5].ndcg@10', 'metric')}."
            ),
            (
                "The train report for depth "
                f"{cite(report, 'retriever.candidate_k', 'count')} truncated "
                f"{cite(report, 'retriever.rerank_truncated_pairs', 'count')} of "
                f"{cite(report, 'question_count * retriever.candidate_k', 'count')} "
                "query/document pairs. Full-ranking retrieval p95 was "
                f"{cite(report, 'timings.retrieval_p95_ms', 'ms')} ms, rerank p95 was "
                f"{cite(report, 'timings.rerank_p95_ms', 'ms')} ms, and the top-K pipeline "
                f"p95 was {cite(report, 'timings.pipeline_p95_ms', 'ms')} ms."
            ),
        ]
    )
    return grid + "\n" + prose


def _heldout_table(specs: list[tuple[str, str]]) -> str:
    return table(
        ["Setup", "Chunks", "Recall@10", "Recall@100", "nDCG@10", "P@10", "MAP", "MRR"],
        [
            [label, cite(path, "chunk_count", "count"), *ranking_cells(path)]
            for label, path in specs
        ],
    )


def render_scifact_test() -> str:
    stats = "benchmarks/scifact/corpus_stats.json"
    base = "benchmarks/scifact/bm25.json"
    selected = "benchmarks/scifact/bm25-selected.json"
    document = "benchmarks/scifact/bm25-document.json"
    hybrid = "benchmarks/scifact/hybrid-selected.json"
    intro = (
        "Corpus fingerprint `"
        f"{cite(base, 'corpus_sha256', 'prefix16')}` and benchmark fingerprint `"
        f"{cite(base, 'benchmark_sha256', 'prefix16')}` "
        "are the leading digits of those fields in the original chunked report. "
        f"{cite(stats, 'query_count', 'count')} queries, "
        f"{cite(stats, 'qrel_count', 'count')} binary qrels. "
        f"{cite(stats, 'document_count', 'count')} documents. "
        "Word length (`\\S+`) has minimum "
        f"{cite(stats, 'word_length.minimum', 'count')}, median "
        f"{cite(stats, 'word_length.p50', 'count')}, mean "
        f"{cite(stats, 'word_length.mean', 'metric')}, and maximum "
        f"{cite(stats, 'word_length.maximum', 'count')}."
    )
    grid = _heldout_table(
        [
            ("Original chunked", base),
            ("Document-level baseline", document),
            ("Selected chunked", selected),
            ("Selected hybrid", hybrid),
        ]
    )
    prose = (
        "The selected hybrid truncated "
        f"{cite(hybrid, 'retriever.truncated_texts', 'count')} of "
        f"{cite(hybrid, 'chunk_count', 'count')} chunk texts. "
        "Full-ranking retrieval p95 was "
        f"{cite(selected, 'timings.retrieval_p95_ms', 'ms')} ms for selected chunked BM25, "
        f"{cite(document, 'timings.retrieval_p95_ms', 'ms')} ms for document-level BM25, and "
        f"{cite(hybrid, 'timings.retrieval_p95_ms', 'ms')} ms for the hybrid."
    )
    return f"{intro}\n\n{grid}\n\n{prose}"


def render_scifact_rerank() -> str:
    path = "benchmarks/scifact/rerank-selected.json"
    grid = table(
        [
            "Setup",
            "Recall@10",
            "Recall@100",
            "nDCG@10",
            "P@10",
            "MAP",
            "MRR",
            "Candidate Recall@100",
        ],
        [
            [
                "Reranked top " + cite(path, "retriever.candidate_k", "count"),
                *ranking_cells(path),
                cite(path, "candidate_recall_at_100", "metric"),
            ]
        ],
    )
    prose = (
        "The run truncated "
        f"{cite(path, 'retriever.rerank_truncated_pairs', 'count')} of "
        f"{cite(path, 'question_count * retriever.candidate_k', 'count')} "
        "query/document pairs. Full-ranking retrieval p95 was "
        f"{cite(path, 'timings.retrieval_p95_ms', 'ms')} ms, rerank p95 was "
        f"{cite(path, 'timings.rerank_p95_ms', 'ms')} ms, and the top-K pipeline p95 was "
        f"{cite(path, 'timings.pipeline_p95_ms', 'ms')} ms."
    )
    return f"{grid}\n\n{prose}"


def render_nfcorpus_test() -> str:
    stats = "benchmarks/nfcorpus/corpus_stats.json"
    whitespace = "benchmarks/nfcorpus/bm25-whitespace.json"
    document = "benchmarks/nfcorpus/bm25-document.json"
    selected = "benchmarks/nfcorpus/bm25-selected.json"
    hybrid = "benchmarks/nfcorpus/hybrid-selected.json"
    intro = (
        f"{cite(stats, 'document_count', 'count')} documents, "
        f"{cite(stats, 'query_count', 'count')} queries, and "
        f"{cite(stats, 'qrel_count', 'count')} qrels ("
        f"{cite(stats, 'grades[grade=1].qrels', 'count')} grade "
        f"{cite(stats, 'grades[grade=1].grade', 'count')} and "
        f"{cite(stats, 'grades[grade=2].qrels', 'count')} grade "
        f"{cite(stats, 'grades[grade=2].grade', 'count')}). "
        "Word length has minimum "
        f"{cite(stats, 'word_length.minimum', 'count')}, median "
        f"{cite(stats, 'word_length.p50', 'count')}, mean "
        f"{cite(stats, 'word_length.mean', 'metric')}, and maximum "
        f"{cite(stats, 'word_length.maximum', 'count')}."
    )
    grid = _heldout_table(
        [
            ("Original chunked definition", whitespace),
            ("Document-level baseline", document),
            ("Selected chunked", selected),
            ("Selected hybrid", hybrid),
        ]
    )
    prose = (
        "The selected hybrid truncated "
        f"{cite(hybrid, 'retriever.truncated_texts', 'count')} of "
        f"{cite(hybrid, 'chunk_count', 'count')} chunk texts. "
        "Full-ranking retrieval p95 was "
        f"{cite(selected, 'timings.retrieval_p95_ms', 'ms')} ms, "
        f"{cite(document, 'timings.retrieval_p95_ms', 'ms')} ms, and "
        f"{cite(hybrid, 'timings.retrieval_p95_ms', 'ms')} ms for selected chunked BM25, "
        "document-level BM25, and the hybrid."
    )
    return f"{intro}\n\n{grid}\n\n{prose}"


def _exploratory_row(label: str, path: str) -> list[str]:
    return [
        label,
        cite(path, "recall_at_k.10", "metric"),
        cite(path, "recall_at_k.100", "metric"),
        cite(path, "ndcg_at_k.10", "metric"),
        cite(path, "mean_average_precision", "metric"),
        cite(path, "mrr", "metric"),
    ]


def render_exploratory() -> str:
    stem = "benchmarks/scifact/bm25-stem-chunk120-exploratory.json"
    dense = "benchmarks/scifact/dense.json"
    minmax = "benchmarks/scifact/hybrid-minmax.json"
    rrf = "benchmarks/scifact/hybrid-rrf.json"
    weighted = "benchmarks/scifact/hybrid-w075.json"
    rows = [
        _exploratory_row("Whitespace chunked BM25", "benchmarks/scifact/bm25.json"),
        _exploratory_row(
            "Stem, " + cite(stem, "config.chunking.chunk_size", "count") + "-word windows",
            stem,
        ),
        _exploratory_row(
            "Dense MiniLM, " + cite(dense, "config.chunking.chunk_size", "count") + "-word windows",
            dense,
        ),
        _exploratory_row(
            "Hybrid min-max, dense_weight "
            + cite(minmax, "config.retrieval.dense_weight", "weight"),
            minmax,
        ),
        _exploratory_row(
            "Hybrid RRF, rrf_k " + cite(rrf, "config.retrieval.rrf_k", "count"),
            rrf,
        ),
        _exploratory_row(
            "Hybrid min-max, dense_weight "
            + cite(weighted, "config.retrieval.dense_weight", "weight"),
            weighted,
        ),
    ]
    return table(["Setup", "Recall@10", "Recall@100", "nDCG@10", "MAP", "MRR"], rows)


def render_verdict_train() -> str:
    path = "benchmarks/scifact/verdict-train-selection.json"
    rows = []
    candidates = lookup(load_report(path), "candidates")
    if not isinstance(candidates, list):
        raise ValueError("verdict candidates must be a list")
    for index, _candidate in enumerate(candidates):
        prefix = f"candidates[{index}]"
        rows.append(
            [
                cite(path, f"{prefix}.doc_k", "count"),
                cite(path, f"{prefix}.sentence_k", "count"),
                cite(path, f"{prefix}.min_confidence", "threshold"),
                cite(path, f"{prefix}.accuracy", "metric"),
                cite(path, f"{prefix}.macro_f1", "metric"),
                cite(path, f"{prefix}.sentence_precision", "metric"),
                cite(path, f"{prefix}.sentence_recall", "metric"),
                cite(path, f"{prefix}.sentence_f1", "metric"),
            ]
        )
    grid = table(
        [
            "doc_k",
            "sentence_k",
            "min_confidence",
            "Accuracy",
            "Macro-F1",
            "Sentence P",
            "Sentence R",
            "Sentence F1",
        ],
        rows,
    )
    prose = (
        f"`doc_k` {cite(path, 'winner.doc_k', 'count')}, "
        f"`sentence_k` {cite(path, 'winner.sentence_k', 'count')}, "
        f"`min_confidence` {cite(path, 'winner.min_confidence', 'threshold')} "
        "is the frozen policy. Its train macro-F1 is "
        f"{cite(path, 'winner.macro_f1', 'metric')} and its mean sentence F1 is "
        f"{cite(path, 'winner.sentence_f1', 'metric')}. "
        "The majority-SUPPORT baseline on the same train claims has accuracy "
        f"{cite(path, 'baseline.accuracy', 'metric')}, macro-F1 "
        f"{cite(path, 'baseline.macro_f1', 'metric')}, and mean sentence F1 "
        f"{cite(path, 'baseline.sentence_f1', 'metric')}. "
        "The grid scored "
        f"{cite(path, 'grid_nli_pairs', 'count')} sentence/claim pairs in "
        f"{cite(path, 'grid_nli_seconds', 'seconds')} s. The winning depth, scored on its own, "
        f"was {cite(path, 'winner_nli_pairs', 'count')} pairs in "
        f"{cite(path, 'winner_nli_seconds', 'seconds')} s. "
        f"{cite(path, 'winner_truncated_pairs', 'count')} pairs were truncated. "
        "Document-level retrieval p95 was "
        f"{cite(path, 'retrieval_p95_ms', 'ms')} ms."
    )
    return f"{grid}\n\n{prose}"


def render_headline() -> str:
    rows = [
        ("Document-level BM25", "benchmarks/scifact/bm25-document.json"),
        ("Selected BM25", "benchmarks/scifact/bm25-selected.json"),
        ("Hybrid", "benchmarks/scifact/hybrid-selected.json"),
        ("Rerank", "benchmarks/scifact/rerank-selected.json"),
    ]
    return table(
        ["Setup", "nDCG@10", "Recall@10"],
        [
            [label, cite(path, "ndcg_at_k.10", "metric"), cite(path, "recall_at_k.10", "metric")]
            for label, path in rows
        ],
    )


def render_chunk_window() -> str:
    sci_stats = "benchmarks/scifact/corpus_stats.json"
    nf_stats = "benchmarks/nfcorpus/corpus_stats.json"
    train = "benchmarks/scifact/bm25-train-selection.json"
    selected_cfg = "configs/scifact-bm25-selected.yaml"
    document_cfg = "configs/scifact-bm25-document.yaml"
    return (
        "Only "
        f"{cite(sci_stats, 'word_length.longer_than[words=480].documents', 'count')} of "
        f"{cite(sci_stats, 'document_count', 'count')} SciFact documents and "
        f"{cite(nf_stats, 'word_length.longer_than[words=480].documents', 'count')} of "
        f"{cite(nf_stats, 'document_count', 'count')} NFCorpus documents are longer than "
        f"{cite(train, 'winner.chunk_size', 'count')} words, so the selected chunked index "
        "is nearly one chunk per document. A post-hoc ablation, scored after the test "
        "numbers were reported and not used to select a setting, uses that document-level "
        "index with parameters "
        f"{cite(selected_cfg, 'retrieval.k1', 'weight')}/"
        f"{cite(selected_cfg, 'retrieval.b', 'weight')}. Its test nDCG@10 is "
        f"{cite('benchmarks/scifact/bm25-document-k1.5-b0.75.json', 'ndcg_at_k.10', 'metric4')}. "
        "Document-level BM25 at "
        f"{cite(document_cfg, 'retrieval.k1', 'weight')}/"
        f"{cite(document_cfg, 'retrieval.b', 'weight')} is "
        f"{cite('benchmarks/scifact/bm25-document.json', 'ndcg_at_k.10', 'metric')}, and the "
        "parameter change is "
        f"{cite('benchmarks/scifact/document-vs-bm25-k15.json', 'checks[metric=ndcg@10].mean_delta', 'signed')}. "
        "The selected chunked index scores "
        f"{cite('benchmarks/scifact/bm25-selected.json', 'ndcg_at_k.10', 'metric4')}. "
        "Those two differ on "
        f"{cite('benchmarks/scifact/bm25-k15-vs-selected.json', 'count:question_changes[metric=ndcg@10][direction!=unchanged]', 'count')} "
        "of "
        f"{cite('benchmarks/scifact/bm25-document-k1.5-b0.75.json', 'question_count', 'count')} "
        "queries, so chunking adds about nothing on SciFact once the parameters match. "
        "On train the frozen pair tie at "
        f"{cite(train, 'winner.metrics.ndcg@10', 'metric4')} vs "
        f"{cite(train, 'document_level_reference.metrics.ndcg@10', 'metric4')}."
    )


def render_verdict_dev() -> str:
    nli = "benchmarks/scifact/verdict-dev.json"
    baseline = "benchmarks/scifact/verdict-baseline-dev.json"
    intro = (
        f"{cite(nli, 'claim_count', 'count')} labeled claims. Gold counts are SUPPORT "
        f"{cite(nli, 'gold_counts.SUPPORT', 'count')}, CONTRADICT "
        f"{cite(nli, 'gold_counts.CONTRADICT', 'count')}, and NEI "
        f"{cite(nli, 'gold_counts.NEI', 'count')}. "
        "Micro sentence F1 is "
        f"{cite(nli, 'micro_sentence_f1', 'metric')} for the frozen NLI and "
        f"{cite(baseline, 'micro_sentence_f1', 'metric')} for the majority baseline. "
        "Evidence-only sentence F1, the mean over claims with nonempty gold evidence, is "
        f"{cite(nli, 'evidence_only_sentence_f1', 'metric')} vs "
        f"{cite(baseline, 'evidence_only_sentence_f1', 'metric')}. "
        "The per-claim mean in the table below is "
        f"{cite(nli, 'sentence_f1', 'metric')} vs "
        f"{cite(baseline, 'sentence_f1', 'metric')}. "
        "That mean scores a both-empty evidence set as a perfect match, which is why it "
        "sits above the micro and evidence-only figures."
    )
    grid = table(
        ["", "Accuracy", "Macro-F1", "Sentence P", "Sentence R", "Sentence F1"],
        [
            ["Majority SUPPORT", *_verdict_cells(baseline)],
            ["Frozen NLI", *_verdict_cells(nli)],
        ],
    )
    classes = (
        "Class F1 for the frozen NLI is SUPPORT "
        f"{cite(nli, 'class_f1.SUPPORT', 'metric')}, CONTRADICT "
        f"{cite(nli, 'class_f1.CONTRADICT', 'metric')}, and NEI "
        f"{cite(nli, 'class_f1.NEI', 'metric')}. "
        "The majority baseline's class F1 is SUPPORT "
        f"{cite(baseline, 'class_f1.SUPPORT', 'metric')}, CONTRADICT "
        f"{cite(baseline, 'class_f1.CONTRADICT', 'metric')}, and NEI "
        f"{cite(baseline, 'class_f1.NEI', 'metric')}."
    )
    micro = (
        "Micro precision, recall, and F1 are "
        f"{cite(nli, 'micro_sentence_precision', 'metric')}, "
        f"{cite(nli, 'micro_sentence_recall', 'metric')}, and "
        f"{cite(nli, 'micro_sentence_f1', 'metric')} for the frozen NLI, and "
        f"{cite(baseline, 'micro_sentence_precision', 'metric')}, "
        f"{cite(baseline, 'micro_sentence_recall', 'metric')}, and "
        f"{cite(baseline, 'micro_sentence_f1', 'metric')} for the baseline."
    )
    timing = (
        "Document-level retrieval p95 was "
        f"{cite(nli, 'retrieval_p95_ms', 'ms')} ms. The NLI pass scored "
        f"{cite(nli, 'nli_pairs', 'count')} pairs in "
        f"{cite(nli, 'nli_seconds', 'seconds')} s. "
        f"{cite(nli, 'nli_truncated_pairs', 'count')} pairs were truncated."
    )
    return f"{intro}\n\n{grid}\n\n{classes}\n\n{micro}\n\n{timing}"


def _verdict_cells(path: str) -> list[str]:
    return [
        cite(path, "accuracy", "metric"),
        cite(path, "macro_f1", "metric"),
        cite(path, "sentence_precision", "metric"),
        cite(path, "sentence_recall", "metric"),
        cite(path, "sentence_f1", "metric"),
    ]


def render_verdict_counts() -> str:
    path = "benchmarks/scifact/verdict-train-selection.json"
    corpus = "benchmarks/scifact/corpus_stats.json"
    return (
        "Train has "
        f"{cite(path, 'mapping.train_qrel_equals_cited', 'count')} claims: the qrel set matches "
        "the evidence documents for "
        f"{cite(path, 'mapping.train_qrel_equals_evidence', 'count')} of them and differs for the "
        f"other {cite(path, 'mapping.train_qrel_equals_cited - mapping.train_qrel_equals_evidence', 'count')}. "
        "Dev has "
        f"{cite(path, 'mapping.dev_qrel_equals_cited', 'count')} claims: "
        f"{cite(path, 'mapping.dev_qrel_equals_evidence', 'count')} match and "
        f"{cite(path, 'mapping.dev_qrel_equals_cited - mapping.dev_qrel_equals_evidence', 'count')} differ. "
        f"{cite(path, 'mapping.exact_space_join_mismatches', 'count')} of "
        f"{cite(corpus, 'document_count', 'count')} abstracts differ from a single-space join "
        "before whitespace is collapsed. Train label counts are SUPPORT "
        f"{cite(path, 'baseline_train_counts.SUPPORT', 'count')}, CONTRADICT "
        f"{cite(path, 'baseline_train_counts.CONTRADICT', 'count')}, and NEI "
        f"{cite(path, 'baseline_train_counts.NEI', 'count')}."
    )


def render_paired_settings() -> str:
    recipe = "configs/paired-uncertainty.yaml"
    verdict = "benchmarks/scifact/verdict-baseline-vs-nli.json"
    pairs = (
        ("statistics.seed", "seed"),
        ("statistics.bootstrap_samples", "bootstrap_samples"),
        ("statistics.permutation_samples", "permutation_samples"),
        ("statistics.confidence", "confidence"),
    )
    for recipe_field, verdict_field in pairs:
        if lookup(load_report(recipe), recipe_field) != lookup(load_report(verdict), verdict_field):
            raise ValueError(f"{recipe} {recipe_field} does not match {verdict} {verdict_field}")
    return (
        "Paired deltas use seed "
        f"{cite(verdict, 'seed', 'count')}, "
        f"{cite(verdict, 'bootstrap_samples', 'count')} bootstrap resamples, "
        f"{cite(verdict, 'permutation_samples', 'count')} sign-flips, and a "
        f"{cite(verdict, 'confidence', 'percent')}% percentile interval. "
        "Committed retrieval comparison files do not store that recipe. "
        "A newly written comparison does, when its thresholds set statistics. "
        "The recipe matches `configs/paired-uncertainty.yaml` and the verdict comparison."
    )


def render_verdict_oracle() -> str:
    path = "benchmarks/scifact/verdict-dev-analysis.json"
    grid = table(
        ["Slice", "Count"],
        [
            ["SUPPORT predicted NEI", cite(path, "errors.support_to_nei", "count")],
            [
                "of those, gold document inside top " + cite(path, "doc_k", "count"),
                cite(path, "errors.support_to_nei_evidence_at_rank_1", "count"),
            ],
            [
                "of those, gold document missed",
                cite(path, "errors.support_to_nei_evidence_missed", "count"),
            ],
            [
                "Incorrect claims with a gold document in the top " + cite(path, "doc_k", "count"),
                cite(path, "errors.incorrect_with_gold_hit", "count"),
            ],
            [
                "Incorrect claims whose gold document was missed",
                cite(path, "errors.incorrect_with_gold_miss", "count"),
            ],
            [
                "Incorrect claims with no gold document",
                cite(path, "errors.incorrect_without_gold_evidence", "count"),
            ],
        ],
    )
    prose = (
        "The oracle gives the frozen sentence policy every gold evidence document and does "
        "not change sentence_k or min_confidence. It was not selected on dev. Its macro-F1 is "
        f"{cite(path, 'oracle_retrieval.macro_f1', 'metric')} and its micro sentence F1 is "
        f"{cite(path, 'oracle_retrieval.micro_sentence_f1', 'metric')}. Evidence-only sentence "
        f"F1 is {cite(path, 'oracle_retrieval.evidence_only_sentence_f1', 'metric')}. "
        "The frozen retrieved NLI system, on the same claims, has macro-F1 "
        f"{cite(path, 'frozen_nli.macro_f1', 'metric')} and micro sentence F1 "
        f"{cite(path, 'frozen_nli.micro_sentence_f1', 'metric')}."
    )
    return grid + "\n\n" + prose


def render_judge_sample() -> str:
    path = "benchmarks/support/judge-sample.json"
    return (
        "The report's own totals are "
        f"{cite(path, 'input_tokens', 'count')} input tokens, "
        f"{cite(path, 'output_tokens', 'count')} output tokens, estimated cost $"
        f"{cite(path, 'estimated_cost_usd', 'usd')}, mean correctness "
        f"{cite(path, 'judge_metrics.judge_correctness', 'metric')}, and mean faithfulness "
        f"{cite(path, 'judge_metrics.judge_faithfulness', 'metric')}."
    )


def _comparison(path: str) -> str:
    return comparison_table(path) + "\n\n" + interval_notes(path)


def _ndcg_differs(path: str, report: str) -> str:
    return (
        "Per-query nDCG@10 differs on "
        + cite(
            path,
            "count:question_changes[metric=ndcg@10][direction!=unchanged]",
            "count",
        )
        + " of "
        + cite(report, "question_count", "count")
        + " queries."
    )


def render_scifact_doc_vs_k15() -> str:
    return (
        _comparison("benchmarks/scifact/document-vs-bm25-k15.json")
        + "\n\n"
        + _ndcg_differs(
            "benchmarks/scifact/document-vs-bm25-k15.json",
            "benchmarks/scifact/bm25-document-k1.5-b0.75.json",
        )
    )


def render_scifact_k15_vs_selected() -> str:
    return (
        _comparison("benchmarks/scifact/bm25-k15-vs-selected.json")
        + "\n\n"
        + _ndcg_differs(
            "benchmarks/scifact/bm25-k15-vs-selected.json",
            "benchmarks/scifact/bm25-document-k1.5-b0.75.json",
        )
    )


def render_nfcorpus_doc_vs_k15() -> str:
    return (
        _comparison("benchmarks/nfcorpus/document-vs-bm25-k15.json")
        + "\n\n"
        + _ndcg_differs(
            "benchmarks/nfcorpus/document-vs-bm25-k15.json",
            "benchmarks/nfcorpus/bm25-document-k1.5-b0.75.json",
        )
    )


def render_nfcorpus_k15_vs_selected() -> str:
    return (
        _comparison("benchmarks/nfcorpus/bm25-k15-vs-selected.json")
        + "\n\n"
        + _ndcg_differs(
            "benchmarks/nfcorpus/bm25-k15-vs-selected.json",
            "benchmarks/nfcorpus/bm25-document-k1.5-b0.75.json",
        )
    )


RENDERERS: dict[str, Callable[[], str]] = {
    "at-a-glance": render_at_a_glance,
    "headline": render_headline,
    "support-fixture": render_support_fixture,
    "chunk-window": render_chunk_window,
    "lexical-train": render_lexical_train,
    "hybrid-weights": render_hybrid_weights,
    "rerank-train": render_rerank_train,
    "scifact-test": render_scifact_test,
    "scifact-doc-vs-selected": lambda: _comparison(
        "benchmarks/scifact/document-vs-bm25-selected.json"
    ),
    "scifact-doc-vs-k15": render_scifact_doc_vs_k15,
    "scifact-k15-vs-selected": render_scifact_k15_vs_selected,
    "scifact-doc-vs-hybrid": lambda: _comparison(
        "benchmarks/scifact/document-vs-hybrid-selected.json"
    ),
    "scifact-selected-vs-hybrid": lambda: _comparison(
        "benchmarks/scifact/bm25-selected-vs-hybrid.json"
    ),
    "scifact-rerank": render_scifact_rerank,
    "scifact-hybrid-vs-rerank": lambda: _comparison("benchmarks/scifact/hybrid-vs-rerank.json"),
    "nfcorpus-test": render_nfcorpus_test,
    "nfcorpus-doc-vs-selected": lambda: _comparison(
        "benchmarks/nfcorpus/document-vs-bm25-selected.json"
    ),
    "nfcorpus-doc-vs-k15": render_nfcorpus_doc_vs_k15,
    "nfcorpus-k15-vs-selected": render_nfcorpus_k15_vs_selected,
    "nfcorpus-doc-vs-hybrid": lambda: _comparison(
        "benchmarks/nfcorpus/document-vs-hybrid-selected.json"
    ),
    "nfcorpus-selected-vs-hybrid": lambda: _comparison(
        "benchmarks/nfcorpus/bm25-selected-vs-hybrid.json"
    ),
    "exploratory": render_exploratory,
    "verdict-counts": render_verdict_counts,
    "verdict-train": render_verdict_train,
    "verdict-dev": render_verdict_dev,
    "verdict-paired": lambda: _comparison("benchmarks/scifact/verdict-baseline-vs-nli.json"),
    "verdict-oracle": render_verdict_oracle,
    "paired-settings": render_paired_settings,
    "judge-sample": render_judge_sample,
}


def render_documents(texts: dict[Path, str]) -> dict[Path, str]:
    """Replace every marked region across the documents. Each region appears once."""
    CITATIONS.clear()
    found: list[str] = []

    def replace(match: re.Match[str]) -> str:
        global CURRENT_REGION
        name = match.group("name")
        if name not in RENDERERS:
            raise ValueError(f"Unknown table region {name}")
        if name in found:
            raise ValueError(f"Table region {name} appears more than once")
        found.append(name)
        CURRENT_REGION = name
        body = RENDERERS[name]().strip("\n")
        return f"<!-- tables:begin {name} -->\n{body}\n<!-- tables:end {name} -->"

    updated = {path: REGION_PATTERN.sub(replace, text) for path, text in texts.items()}
    missing = [name for name in RENDERERS if name not in found]
    if missing:
        raise ValueError("Missing table regions: " + ", ".join(missing))
    return updated


def render_readme(text: str) -> str:
    """Backward-compatible helper. Prefer :func:`render_documents`."""
    return render_documents({README: text})[README]


def strip_table_headers(region: str) -> str:
    lines = region.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        header = lines[index].startswith("| ")
        separator = index + 1 < len(lines) and lines[index + 1].startswith("| ---")
        if header and separator:
            index += 2
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept)


def uncited_digits(region: str, shown: list[str]) -> str:
    """Digits left after cited displays and metric labels are removed."""
    text = strip_table_headers(region)
    for fragment in sorted(LABEL_FRAGMENTS, key=len, reverse=True):
        text = text.replace(fragment, "")
    for token in sorted(set(shown), key=len, reverse=True):
        text = text.replace(token, "")
    return "".join(character for character in text if character.isdigit())


def main() -> None:
    originals = {path: path.read_text(encoding="utf-8") for path in DOCUMENTS}
    updated = render_documents(originals)
    changed = False
    for path, text in updated.items():
        if text != originals[path]:
            path.write_text(text, encoding="utf-8")
            print(f"Updated {path}")
            changed = True
    if not changed:
        print("Document regions already match the reports")


if __name__ == "__main__":
    main()
