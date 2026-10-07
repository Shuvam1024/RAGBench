from pathlib import Path
from statistics import fmean

import pytest
from typer.testing import CliRunner

from ragstat.cli import app, save_result
from ragstat.config import load_config
from ragstat.evaluation.comparison import (
    StatisticsConfig,
    Thresholds,
    compare,
    load_report,
    load_thresholds,
)
from ragstat.evaluation.models import EvaluationResult
from ragstat.evaluation.runner import evaluate


@pytest.fixture
def report() -> EvaluationResult:
    return evaluate(load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml"))


def test_saved_reports_still_load_and_compare(report: EvaluationResult) -> None:
    root = Path(__file__).resolve().parents[1]
    saved = load_report(root / "benchmarks" / "baseline.json")
    judged = load_report(root / "benchmarks" / "support" / "judge-sample.json")
    assert saved.versions["ragbench"] == "1.0.0"
    assert "ragstat" not in saved.versions
    assert judged.versions["ragbench"] == "1.1.0"
    assert any(
        question.judge is not None and question.judge.rubric_version == "ragbench-judge-v1"
        for question in judged.questions
    )
    limits = load_thresholds(root / "configs" / "thresholds.yaml")
    assert compare(saved, saved, limits).passed
    assert "ragstat" in report.versions
    assert report.versions["ragstat"] != "not-installed"
    assert "ragbench" not in report.versions
    assert saved.mrr == report.mrr
    assert saved.recall_at_k == report.recall_at_k
    assert compare(saved, report, limits).passed


def test_identical_and_boundary(report: EvaluationResult) -> None:
    assert compare(report, report, Thresholds()).passed
    questions = tuple(
        q.model_copy(update={"reciprocal_rank": q.reciprocal_rank / 2}) for q in report.questions
    )
    worse = report.model_copy(update={"mrr": report.mrr / 2, "questions": questions})
    assert not compare(report, worse, Thresholds(max_drop={"mrr": 0.0})).passed
    assert compare(report, worse, Thresholds(max_drop={"mrr": report.mrr / 2})).passed


@pytest.mark.parametrize(
    "field,value",
    [("corpus_sha256", "different"), ("benchmark_sha256", "different"), ("question_count", 500)],
)
def test_incompatible(report: EvaluationResult, field: str, value: object) -> None:
    with pytest.raises(ValueError, match="Incompatible"):
        compare(report, report.model_copy(update={field: value}), Thresholds())


def test_unknown_missing_and_empty_metrics(report: EvaluationResult) -> None:
    for raw in (
        {"max_drop": {"unknown": 0.1}},
        {"max_drop": {}, "max_increase_ratio": {}},
        {"max_drop": {"mrr": True}},
    ):
        with pytest.raises(ValueError):
            Thresholds.model_validate(raw)
    with pytest.raises(ValueError, match="no finite"):
        compare(report, report, Thresholds(max_drop={"answer_token_f1": 0.0}))


def test_resource_zero_baseline_and_environment(report: EvaluationResult) -> None:
    old = report.model_copy(update={"estimated_cost_usd": 0.0})
    new = old.model_copy(update={"estimated_cost_usd": 0.01})
    limits = Thresholds(max_drop={}, max_increase_ratio={"estimated_cost_usd": 0.2})
    assert compare(old, old, limits).passed
    assert not compare(old, new, limits).passed
    with pytest.raises(ValueError, match="environment"):
        compare(
            old,
            new.model_copy(update={"environment": {}}),
            Thresholds(max_drop={}, max_increase_ratio={"retrieval_p95_ms": 0.5}),
        )


def test_question_changes_and_ci_gate(report: EvaluationResult) -> None:
    base_rr = [question.reciprocal_rank for question in report.questions]
    index = next(position for position, value in enumerate(base_rr) if value > 0)
    candidate_rr = base_rr.copy()
    candidate_rr[index] = 0.0
    questions = tuple(
        question.model_copy(update={"reciprocal_rank": value})
        for question, value in zip(report.questions, candidate_rr, strict=True)
    )
    candidate = report.model_copy(update={"mrr": fmean(candidate_rr), "questions": questions})
    statistics = StatisticsConfig(
        seed=0, bootstrap_samples=400, permutation_samples=200, gate_on_ci=False
    )
    point = compare(report, candidate, Thresholds(max_drop={"mrr": 0.0}, statistics=statistics))
    assert not point.passed
    assert point.checks[0].ci_high == pytest.approx(0.0)
    assert point.statistics is not None
    assert point.statistics.seed == 0
    assert point.statistics.bootstrap_samples == 400
    assert point.statistics.permutation_samples == 200
    assert point.statistics.confidence == pytest.approx(0.95)
    assert point.statistics.gate_mode == "point_drop"
    assert compare(report, candidate, Thresholds(max_drop={"mrr": 0.0})).statistics is None
    gated = compare(
        report,
        candidate,
        Thresholds(
            max_drop={"mrr": 0.0}, statistics=statistics.model_copy(update={"gate_on_ci": True})
        ),
    )
    assert gated.passed
    assert gated.checks[0].rule.startswith("proven_regression:")
    non_inferior = compare(
        report,
        candidate,
        Thresholds(
            max_drop={"mrr": 0.0},
            statistics=statistics.model_copy(update={"gate_mode": "non_inferior"}),
        ),
    )
    assert not non_inferior.passed
    assert non_inferior.checks[0].ci_low is not None
    assert non_inferior.checks[0].ci_low < 0
    assert non_inferior.checks[0].rule.startswith("non_inferior:")
    wide_margin = compare(
        report,
        candidate,
        Thresholds(
            max_drop={"mrr": 1.0},
            statistics=statistics.model_copy(update={"gate_mode": "non_inferior"}),
        ),
    )
    assert wide_margin.passed
    regressed = [item for item in gated.question_changes if item.direction == "regressed"]
    assert [item.question_id for item in regressed] == [report.questions[index].id]
    again = compare(report, candidate, Thresholds(max_drop={"mrr": 0.0}, statistics=statistics))
    assert [(item.ci_low, item.ci_high, item.permutation_p_value) for item in point.checks] == [
        (item.ci_low, item.ci_high, item.permutation_p_value) for item in again.checks
    ]


def test_cli_exit_codes_and_corrupt_report(report: EvaluationResult, tmp_path: Path) -> None:
    baseline, candidate, thresholds = (
        tmp_path / name for name in ("base.json", "new.json", "limits.yaml")
    )
    save_result(report, baseline)
    save_result(report, candidate)
    thresholds.write_text("max_drop: {mrr: 0.0}\n")
    args = [
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--thresholds",
        str(thresholds),
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    questions = tuple(q.model_copy(update={"reciprocal_rank": 0.0}) for q in report.questions)
    save_result(report.model_copy(update={"mrr": 0.0, "questions": questions}), candidate)
    assert CliRunner().invoke(app, args).exit_code == 2
    candidate.write_text('{"schema_version": 1}')
    assert CliRunner().invoke(app, args).exit_code == 1
    with pytest.raises(ValueError):
        load_report(candidate)


def test_gate_mode_names_and_committed_ci_policy() -> None:
    legacy = StatisticsConfig(gate_on_ci=True)
    assert legacy.resolved_gate_mode() == "proven_regression"
    named = StatisticsConfig(gate_mode="non_inferior")
    assert named.resolved_gate_mode() == "non_inferior"
    with pytest.raises(ValueError, match="conflicts"):
        StatisticsConfig(gate_on_ci=True, gate_mode="non_inferior")
    with pytest.raises(ValueError):
        StatisticsConfig.model_validate({"gate_mode": "upper_bound"})
    root = Path(__file__).resolve().parents[1]
    for name in ("thresholds.yaml", "scifact-thresholds.yaml"):
        limits = load_thresholds(root / "configs" / name)
        assert limits.statistics is not None
        assert limits.statistics.gate_on_ci is False
        assert limits.statistics.resolved_gate_mode() == "point_drop"
