"""Show the support-fixture gate reject a 0.07 quality drop.

The process exits 0 when that rejection happens. ``ragbench compare`` on the
written candidate exits 2. Set ``GITHUB_STEP_SUMMARY`` to append the metric
table to a GitHub Actions job summary.
"""

import os
from pathlib import Path

from ragbench.cli import save_result
from ragbench.evaluation.comparison import load_report, load_thresholds
from ragbench.evaluation.gate_demo import degrade_support_metrics, negative_control_summary

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline.json"
THRESHOLDS = ROOT / "configs" / "thresholds.yaml"


def main(write_candidate: Path | None = None) -> int:
    baseline = load_report(BASELINE)
    summary, control, policies = negative_control_summary(baseline, load_thresholds(THRESHOLDS))
    text = summary + (
        "\nThe negative-control table is the CI policy (`point_drop`). "
        "A passing `proven_regression` row in the illustration means the upper "
        "bound did not establish a regression. A failing `non_inferior` row means "
        "the lower bound did not clear the margin.\n"
    )
    print(text)
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as handle:
            handle.write(text)
    if write_candidate is not None:
        save_result(degrade_support_metrics(baseline), write_candidate)
        print(f"Wrote degraded candidate: {write_candidate}")
    expected = (
        not control.passed
        and policies["point_drop"].passed is False
        and policies["proven_regression"].passed is True
        and policies["non_inferior"].passed is False
    )
    if not expected:
        print("Negative control did not match the expected gate decisions.", flush=True)
        return 1
    print(
        "Negative control rejected the 0.07 drop, as required. "
        "ragbench compare on this candidate exits 2."
    )
    return 0


if __name__ == "__main__":
    import sys

    candidate = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(candidate))
