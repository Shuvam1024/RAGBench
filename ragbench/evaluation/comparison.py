"""Fail-closed regression checks on compatible, validated evaluation reports."""

import math
import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from ragbench.config import ConfigModel, NonnegativeFloat, UnitFloat
from ragbench.evaluation.models import EvaluationResult, Record

QUALITY = {
    "mrr",
    "answer_exact_match",
    "answer_token_f1",
    "context_token_precision",
    "judge_correctness",
    "judge_faithfulness",
}
RESOURCES = {"retrieval_p95_ms", "generation_p95_ms", "estimated_cost_usd"}


class Thresholds(ConfigModel):
    max_drop: dict[str, UnitFloat] = Field(default_factory=lambda: {"mrr": 0.0, "recall@1": 0.0})
    max_increase_ratio: dict[str, NonnegativeFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_metrics(self) -> Self:
        if not self.max_drop and not self.max_increase_ratio:
            raise ValueError("At least one regression threshold is required")
        for name in self.max_drop:
            if name not in QUALITY and not re.fullmatch(r"recall@[1-9][0-9]*", name):
                raise ValueError(f"Unknown quality metric: {name}")
        if set(self.max_increase_ratio) - RESOURCES:
            raise ValueError("Unknown resource metric in max_increase_ratio")
        return self


class MetricCheck(Record):
    metric: str
    baseline: float
    candidate: float
    tolerance: float
    change: float | None
    passed: bool
    rule: str


class Comparison(Record):
    baseline_run_id: str
    candidate_run_id: str
    passed: bool
    checks: tuple[MetricCheck, ...]


def load_report(path: Path) -> EvaluationResult:
    try:
        return EvaluationResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load version 2 report {path}: {exc}") from exc


def load_thresholds(path: Path) -> Thresholds:
    try:
        return Thresholds.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid thresholds {path}: {exc}") from exc


def metric(report: EvaluationResult, name: str) -> float:
    value: float | None
    if name == "mrr":
        value = report.mrr
    elif name.startswith("recall@"):
        value = report.recall_at_k.get(int(name.split("@")[1]))
    elif name in {"retrieval_p95_ms", "generation_p95_ms"}:
        value = getattr(report.timings, name)
    elif name == "estimated_cost_usd":
        value = report.estimated_cost_usd
    elif name.startswith("judge_"):
        value = (report.judge_metrics or {}).get(name)
    else:
        value = (report.answer_metrics or {}).get(name)
    if value is None or not math.isfinite(value):
        raise ValueError(f"Report {report.run_id} has no finite measurement for {name}")
    return value


def compare(
    baseline: EvaluationResult, candidate: EvaluationResult, limits: Thresholds
) -> Comparison:
    for field in ("schema_version", "corpus_sha256", "benchmark_sha256", "question_count"):
        if getattr(baseline, field) != getattr(candidate, field):
            raise ValueError(f"Incompatible reports: {field} differs")
    if {q.id for q in baseline.questions} != {q.id for q in candidate.questions}:
        raise ValueError("Incompatible reports: question IDs differ")
    if any(name.endswith("_ms") for name in limits.max_increase_ratio):
        if baseline.environment != candidate.environment:
            raise ValueError("Latency gates require matching environment metadata")
    if any(name.startswith("judge_") for name in limits.max_drop):
        if baseline.config.judge != candidate.config.judge:
            raise ValueError("Judge gates require the same judge configuration")
        identities = [
            {
                (q.judge.rubric_version, q.judge.response.model)
                for q in report.questions
                if q.judge is not None
            }
            for report in (baseline, candidate)
        ]
        if len(identities[0]) != 1 or identities[0] != identities[1]:
            raise ValueError("Judge gates require the same resolved model and rubric")
    checks: list[MetricCheck] = []
    for name, tolerance in limits.max_drop.items():
        old, new = metric(baseline, name), metric(candidate, name)
        drop = old - new
        passed = drop <= tolerance or math.isclose(drop, tolerance, rel_tol=0, abs_tol=1e-12)
        checks.append(
            MetricCheck(
                metric=name,
                baseline=old,
                candidate=new,
                tolerance=tolerance,
                change=drop,
                passed=passed,
                rule="maximum absolute drop",
            )
        )
    for name, tolerance in limits.max_increase_ratio.items():
        old, new = metric(baseline, name), metric(candidate, name)
        change = (new - old) / old if old else (0.0 if new == 0 else None)
        passed = change is not None and (
            change <= tolerance or math.isclose(change, tolerance, abs_tol=1e-12)
        )
        checks.append(
            MetricCheck(
                metric=name,
                baseline=old,
                candidate=new,
                tolerance=tolerance,
                change=change,
                passed=passed,
                rule="maximum relative increase",
            )
        )
    return Comparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        passed=all(item.passed for item in checks),
        checks=tuple(checks),
    )
