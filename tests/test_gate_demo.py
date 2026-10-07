from pathlib import Path
from statistics import fmean

import pytest
from typer.testing import CliRunner

from ragbench.cli import app, save_result
from ragbench.evaluation.comparison import load_report, load_thresholds
from ragbench.evaluation.gate_demo import (
    NEGATIVE_CONTROL_DROP,
    degrade_support_metrics,
    negative_control_summary,
    reduce_values,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline.json"
THRESHOLDS = ROOT / "configs" / "thresholds.yaml"


def test_reduce_values_lowers_the_mean_from_the_largest_entries() -> None:
    updated = reduce_values([1.0, 1.0, 0.0], 0.5)
    assert updated == pytest.approx([0.0, 0.5, 0.0])
    assert fmean(updated) == pytest.approx(fmean([1.0, 1.0, 0.0]) - 0.5)
    with pytest.raises(ValueError, match="cannot lower"):
        reduce_values([0.1, 0.1], 0.5)


def test_negative_control_fails_point_drop_and_splits_interval_policies() -> None:
    baseline = load_report(BASELINE)
    degraded = degrade_support_metrics(baseline)
    assert degraded.mrr == pytest.approx(baseline.mrr - NEGATIVE_CONTROL_DROP)
    assert degraded.recall_at_k[1] == pytest.approx(baseline.recall_at_k[1] - NEGATIVE_CONTROL_DROP)
    assert degraded.recall_at_k[3] == pytest.approx(baseline.recall_at_k[3] - NEGATIVE_CONTROL_DROP)
    summary, control, policies = negative_control_summary(baseline, load_thresholds(THRESHOLDS))
    assert not control.passed
    assert {check.metric for check in control.checks if not check.passed} >= {
        "mrr",
        "recall@1",
        "recall@3",
    }
    assert policies["point_drop"].passed is False
    assert policies["proven_regression"].passed is True
    assert policies["non_inferior"].passed is False
    assert "Negative control: support-fixture gate" in summary
    assert "FAIL" in summary
    assert "`proven_regression` would PASS" in summary
    assert "`non_inferior` would FAIL" in summary


def test_degraded_candidate_exits_2(tmp_path: Path) -> None:
    candidate = tmp_path / "degraded.json"
    save_result(degrade_support_metrics(load_report(BASELINE)), candidate)
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(candidate),
            "--thresholds",
            str(THRESHOLDS),
        ],
    )
    assert result.exit_code == 2
    assert "FAIL" in result.output
    assert "Regression detected" in result.output
