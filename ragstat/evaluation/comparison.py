"""Fail-closed regression checks on compatible, validated evaluation reports."""

import math
import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from ragstat.config import ConfigModel, NonnegativeFloat, PositiveInt, UnitFloat
from ragstat.evaluation.models import EvaluationResult, QuestionResult, Record
from ragstat.evaluation.statistics import bootstrap_mean_ci, signflip_p_value

QUALITY = {
    "mrr",
    "map",
    "answer_exact_match",
    "answer_token_f1",
    "context_token_precision",
    "judge_correctness",
    "judge_faithfulness",
}
RESOURCES = {"retrieval_p95_ms", "generation_p95_ms", "estimated_cost_usd"}
CUTOFF_METRIC = re.compile(r"(recall|ndcg|precision)@[1-9][0-9]*")
GateMode = Literal["point_drop", "proven_regression", "non_inferior"]


class StatisticsConfig(ConfigModel):
    """Paired uncertainty over the benchmark's questions. Off unless configured.

    ``gate_mode`` names the quality decision. ``point_drop`` uses the point
    estimate. ``proven_regression`` fails only when the upper confidence bound
    of ``(candidate - baseline)`` is below ``-tolerance``. ``non_inferior``
    fails unless the lower bound is at least ``-tolerance``. ``gate_on_ci``
    remains the older switch: true means ``proven_regression`` when
    ``gate_mode`` is omitted.
    """

    seed: int = Field(default=0, ge=0)
    bootstrap_samples: PositiveInt = 10_000
    permutation_samples: PositiveInt = 10_000
    confidence: float = Field(gt=0, lt=1, default=0.95)
    gate_on_ci: bool = False
    gate_mode: GateMode | None = None

    @model_validator(mode="after")
    def legacy_flag_matches_named_mode(self) -> Self:
        if self.gate_on_ci and self.gate_mode not in (None, "proven_regression"):
            raise ValueError(
                "gate_on_ci true selects proven_regression and conflicts with "
                f"gate_mode {self.gate_mode!r}"
            )
        return self

    def resolved_gate_mode(self) -> GateMode:
        if self.gate_mode is not None:
            return self.gate_mode
        return "proven_regression" if self.gate_on_ci else "point_drop"


class Thresholds(ConfigModel):
    max_drop: dict[str, UnitFloat] = Field(default_factory=lambda: {"mrr": 0.0, "recall@1": 0.0})
    max_increase_ratio: dict[str, NonnegativeFloat] = Field(default_factory=dict)
    statistics: StatisticsConfig | None = None

    @model_validator(mode="after")
    def known_metrics(self) -> Self:
        if not self.max_drop and not self.max_increase_ratio:
            raise ValueError("At least one regression threshold is required")
        for name in self.max_drop:
            if name not in QUALITY and CUTOFF_METRIC.fullmatch(name) is None:
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
    mean_delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    permutation_p_value: float | None = None


class QuestionChange(Record):
    question_id: str
    metric: str
    baseline: float
    candidate: float
    delta: float
    direction: Literal["improved", "regressed", "unchanged"]


class ComparisonStatistics(Record):
    """The uncertainty recipe that produced a comparison file."""

    seed: int
    bootstrap_samples: int
    permutation_samples: int
    confidence: float
    gate_mode: GateMode


class Comparison(Record):
    baseline_run_id: str
    candidate_run_id: str
    passed: bool
    checks: tuple[MetricCheck, ...]
    question_changes: tuple[QuestionChange, ...] = ()
    statistics: ComparisonStatistics | None = None


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
    elif name == "map":
        value = report.mean_average_precision
    elif name.startswith("recall@"):
        value = report.recall_at_k.get(int(name.split("@")[1]))
    elif name.startswith("ndcg@"):
        value = None if report.ndcg_at_k is None else report.ndcg_at_k.get(int(name.split("@")[1]))
    elif name.startswith("precision@"):
        value = (
            None
            if report.precision_at_k is None
            else report.precision_at_k.get(int(name.split("@")[1]))
        )
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


def question_metric(question: QuestionResult, name: str) -> float:
    value: float | None
    if name == "mrr":
        value = question.reciprocal_rank
    elif name == "map":
        value = question.average_precision
    elif name.startswith("recall@"):
        value = question.recall_at_k.get(int(name.split("@")[1]))
    elif name.startswith("ndcg@"):
        value = (
            None if question.ndcg_at_k is None else question.ndcg_at_k.get(int(name.split("@")[1]))
        )
    elif name.startswith("precision@"):
        value = (
            None
            if question.precision_at_k is None
            else question.precision_at_k.get(int(name.split("@")[1]))
        )
    elif name.startswith("judge_"):
        value = (
            None
            if question.judge is None
            else {
                "judge_correctness": question.judge.scores.correctness / 4,
                "judge_faithfulness": question.judge.scores.faithfulness / 4,
            }.get(name)
        )
    else:
        value = None if question.answer_metrics is None else question.answer_metrics.get(name)
    if value is None or not math.isfinite(value):
        raise ValueError(f"Question {question.id} has no finite measurement for {name}")
    return value


def _direction(delta: float) -> Literal["improved", "regressed", "unchanged"]:
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        return "unchanged"
    return "improved" if delta > 0 else "regressed"


def _statistics_fields(
    name: str,
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    settings: StatisticsConfig,
) -> tuple[float, float, float, float]:
    by_id = {question.id: question for question in candidate.questions}
    deltas = [
        question_metric(by_id[question.id], name) - question_metric(question, name)
        for question in baseline.questions
    ]
    mean_delta, low, high = bootstrap_mean_ci(
        deltas,
        samples=settings.bootstrap_samples,
        seed=f"ragbench-stats-v1:{settings.seed}:bootstrap:{name}",
        confidence=settings.confidence,
    )
    p_value = signflip_p_value(
        deltas,
        samples=settings.permutation_samples,
        seed=f"ragbench-stats-v1:{settings.seed}:permutation:{name}",
    )
    return mean_delta, low, high, p_value


def decide_quality(
    mode: GateMode,
    *,
    point_passed: bool,
    tolerance: float,
    ci_low: float | None,
    ci_high: float | None,
) -> tuple[bool, str]:
    """Apply one named quality policy. ``tolerance`` is the allowed negative margin."""
    if mode == "proven_regression":
        if ci_high is None:
            raise ValueError("proven_regression requires a confidence interval")
        passed = ci_high > -tolerance or math.isclose(ci_high, -tolerance, abs_tol=1e-12)
        rule = (
            "proven_regression: upper confidence bound of (candidate - baseline) "
            "is at least -tolerance"
        )
    elif mode == "non_inferior":
        if ci_low is None:
            raise ValueError("non_inferior requires a confidence interval")
        passed = ci_low > -tolerance or math.isclose(ci_low, -tolerance, abs_tol=1e-12)
        rule = (
            "non_inferior: lower confidence bound of (candidate - baseline) is at least -tolerance"
        )
    elif mode == "point_drop":
        passed = point_passed
        rule = "point_drop: maximum absolute drop"
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    return passed, rule


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
    changes: list[QuestionChange] = []
    candidate_by_id = {question.id: question for question in candidate.questions}
    for name, tolerance in limits.max_drop.items():
        old, new = metric(baseline, name), metric(candidate, name)
        drop = old - new
        point_passed = drop <= tolerance or math.isclose(drop, tolerance, rel_tol=0, abs_tol=1e-12)
        mean_delta = ci_low = ci_high = p_value = None
        mode: GateMode = "point_drop"
        if limits.statistics is not None:
            mean_delta, ci_low, ci_high, p_value = _statistics_fields(
                name, baseline, candidate, limits.statistics
            )
            mode = limits.statistics.resolved_gate_mode()
        passed, rule = decide_quality(
            mode,
            point_passed=point_passed,
            tolerance=tolerance,
            ci_low=ci_low,
            ci_high=ci_high,
        )
        checks.append(
            MetricCheck(
                metric=name,
                baseline=old,
                candidate=new,
                tolerance=tolerance,
                change=drop,
                passed=passed,
                rule=rule,
                mean_delta=mean_delta,
                ci_low=ci_low,
                ci_high=ci_high,
                permutation_p_value=p_value,
            )
        )
        for question in baseline.questions:
            base_value = question_metric(question, name)
            cand_value = question_metric(candidate_by_id[question.id], name)
            delta = cand_value - base_value
            changes.append(
                QuestionChange(
                    question_id=question.id,
                    metric=name,
                    baseline=base_value,
                    candidate=cand_value,
                    delta=delta,
                    direction=_direction(delta),
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
                rule="point_drop: maximum relative increase",
            )
        )
    recipe = None
    if limits.statistics is not None:
        recipe = ComparisonStatistics(
            seed=limits.statistics.seed,
            bootstrap_samples=limits.statistics.bootstrap_samples,
            permutation_samples=limits.statistics.permutation_samples,
            confidence=limits.statistics.confidence,
            gate_mode=limits.statistics.resolved_gate_mode(),
        )
    return Comparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        passed=all(item.passed for item in checks),
        checks=tuple(checks),
        question_changes=tuple(sorted(changes, key=lambda item: (item.metric, item.question_id))),
        statistics=recipe,
    )
