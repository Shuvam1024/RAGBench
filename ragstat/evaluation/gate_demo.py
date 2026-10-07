"""Deliberate regressions used to show the quality gate reject a candidate.

The support-fixture thresholds allow a 0.02 point drop. Lowering MRR,
Recall@1, and Recall@3 by 0.07 must fail that gate. A separate one-question
change shows why ``proven_regression`` and ``non_inferior`` are different
decisions on a 27-question sample.
"""

from collections.abc import Sequence
from statistics import fmean

from ragstat.evaluation.comparison import (
    Comparison,
    StatisticsConfig,
    Thresholds,
    compare,
    decide_quality,
)
from ragstat.evaluation.models import EvaluationResult

NEGATIVE_CONTROL_DROP = 0.07
_MASS_TOLERANCE = 1e-9


def reduce_values(values: Sequence[float], amount: float) -> list[float]:
    """Lower a sample's mean by ``amount``, taking mass from the largest values."""
    if isinstance(amount, bool) or not isinstance(amount, int | float) or amount < 0:
        raise ValueError("drop amount must be a nonnegative real number")
    if not values:
        raise ValueError("drop requires at least one value")
    remaining = float(amount) * len(values)
    updated = [float(value) for value in values]
    for index in sorted(range(len(updated)), key=lambda item: updated[item], reverse=True):
        take = min(updated[index], remaining)
        updated[index] -= take
        remaining -= take
        if remaining <= 1e-12:
            remaining = 0.0
            break
    if remaining > _MASS_TOLERANCE:
        raise ValueError(f"cannot lower the mean by {amount}")
    return updated


def degrade_support_metrics(
    report: EvaluationResult, amount: float = NEGATIVE_CONTROL_DROP
) -> EvaluationResult:
    """Return a compatible report whose MRR, Recall@1, and Recall@3 means fall by ``amount``."""
    reciprocal = reduce_values([question.reciprocal_rank for question in report.questions], amount)
    recall_1 = reduce_values([question.recall_at_k[1] for question in report.questions], amount)
    recall_3 = reduce_values([question.recall_at_k[3] for question in report.questions], amount)
    questions = tuple(
        question.model_copy(
            update={
                "reciprocal_rank": reciprocal[index],
                "recall_at_k": {**question.recall_at_k, 1: recall_1[index], 3: recall_3[index]},
            }
        )
        for index, question in enumerate(report.questions)
    )
    return report.model_copy(
        update={
            "run_id": f"negative-control-drop-{amount:g}",
            "questions": questions,
            "mrr": fmean(reciprocal),
            "recall_at_k": {**report.recall_at_k, 1: fmean(recall_1), 3: fmean(recall_3)},
        }
    )


def zero_one_reciprocal_rank(report: EvaluationResult) -> EvaluationResult:
    """Set one positive reciprocal rank to zero and keep the aggregate MRR aligned."""
    index = next(
        position
        for position, question in enumerate(report.questions)
        if question.reciprocal_rank > 0
    )
    reciprocal = [question.reciprocal_rank for question in report.questions]
    reciprocal[index] = 0.0
    questions = tuple(
        question.model_copy(update={"reciprocal_rank": value})
        for question, value in zip(report.questions, reciprocal, strict=True)
    )
    return report.model_copy(
        update={
            "run_id": "policy-illustration-one-question",
            "questions": questions,
            "mrr": fmean(reciprocal),
        }
    )


def _bound(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def metric_table(result: Comparison) -> str:
    """Markdown table of one comparison, including intervals when they were recorded."""
    lines = [
        "| Metric | Baseline | Candidate | Drop | Mean delta | 95% CI low | 95% CI high | Result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for check in result.checks:
        change = "" if check.change is None else f"{check.change:.4f}"
        lines.append(
            "| {metric} | {baseline:.4f} | {candidate:.4f} | {change} | {delta} | {low} | {high} | {result} |".format(
                metric=check.metric,
                baseline=check.baseline,
                candidate=check.candidate,
                change=change,
                delta=_bound(check.mean_delta),
                low=_bound(check.ci_low),
                high=_bound(check.ci_high),
                result="PASS" if check.passed else "FAIL",
            )
        )
    return "\n".join(lines)


def _policy_row(name: str, result: Comparison) -> str:
    check = result.checks[0]
    return (
        f"| {name} | {check.rule.split(':', 1)[0]} | "
        f"{'PASS' if result.passed else 'FAIL'} | {_bound(check.ci_low)} | {_bound(check.ci_high)} |"
    )


def negative_control_summary(
    baseline: EvaluationResult, limits: Thresholds
) -> tuple[str, Comparison, dict[str, Comparison]]:
    """Build the Actions summary for the expected gate failure and the policy contrast.

    The returned comparison is the support-fixture point-drop check. It is
    expected to fail. The policy map uses a one-question MRR change at margin 0.
    """
    if limits.statistics is None or limits.statistics.resolved_gate_mode() != "point_drop":
        raise ValueError("the negative control demonstrates the point_drop thresholds")
    degraded = degrade_support_metrics(baseline)
    control = compare(baseline, degraded, limits)
    illustration_questions = zero_one_reciprocal_rank(baseline)
    statistics = StatisticsConfig(
        seed=0, bootstrap_samples=10_000, permutation_samples=10_000, confidence=0.95
    )
    policies = {
        mode: compare(
            baseline,
            illustration_questions,
            Thresholds(
                max_drop={"mrr": 0.0}, statistics=statistics.model_copy(update={"gate_mode": mode})
            ),
        )
        for mode in ("point_drop", "proven_regression", "non_inferior")
    }
    question_count = baseline.question_count
    mrr = next(check for check in control.checks if check.metric == "mrr")
    upper_passes, _rule = decide_quality(
        "proven_regression",
        point_passed=False,
        tolerance=mrr.tolerance,
        ci_low=mrr.ci_low,
        ci_high=mrr.ci_high,
    )
    lower_passes, _rule = decide_quality(
        "non_inferior",
        point_passed=False,
        tolerance=mrr.tolerance,
        ci_low=mrr.ci_low,
        ci_high=mrr.ci_high,
    )
    body = "\n".join(
        [
            "## Negative control: support-fixture gate",
            "",
            "This check degrades the committed support-fixture baseline by "
            f"{NEGATIVE_CONTROL_DROP:.2f} on MRR, Recall@1, and Recall@3. "
            "`configs/thresholds.yaml` allows a 0.02 point drop. "
            "`gate_on_ci: false` keeps that file on the `point_drop` policy. "
            "`ragstat compare` is expected to exit 2. This job stays green only "
            "when that rejection happens.",
            "",
            metric_table(control),
            "",
            "The Result column is `point_drop`. On MRR, `proven_regression` would "
            f"{'PASS' if upper_passes else 'FAIL'} and `non_inferior` would "
            f"{'PASS' if lower_passes else 'FAIL'} "
            f"(interval [{_bound(mrr.ci_low)}, {_bound(mrr.ci_high)}], "
            f"margin {mrr.tolerance:.2f}). The drop is concentrated on a few "
            f"questions in this {question_count}-question fixture, so the upper "
            "bound can still reach 0.",
            "",
            "## Policy illustration on one changed question",
            "",
            f"The support fixture has {question_count} questions. One reciprocal rank "
            "is set to 0 and the margin is 0. `point_drop` follows the mean. "
            "`proven_regression` fails only when the upper confidence bound of "
            "(candidate - baseline) is below the margin. `non_inferior` fails unless "
            "the lower bound stays at or above the margin. A 300-query set makes "
            "that interval narrower; this 27-question sample is wide enough that "
            "the two interval policies disagree.",
            "",
            "| Policy | Decision | Result | 95% CI low | 95% CI high |",
            "| --- | --- | --- | --- | --- |",
            _policy_row("point_drop", policies["point_drop"]),
            _policy_row("proven_regression", policies["proven_regression"]),
            _policy_row("non_inferior", policies["non_inferior"]),
            "",
        ]
    )
    return body, control, policies
