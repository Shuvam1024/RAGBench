# Regression testing

A comparison answers whether a candidate's measured behavior stays within the
allowed change from a baseline. It never silently changes that baseline.
Continuous integration gates the support fixture and two frozen SciFact BM25
configs, `configs/scifact-bm25-document.yaml` and
`configs/scifact-bm25-selected.yaml`, with a zero point-drop. The exploratory
chunk-120 file is not that gate. A newly written comparison stores the
statistics recipe when the thresholds file sets one. Committed comparison
files keep the bytes they were saved with.

```bash
ragstat evaluate --config configs/baseline.yaml --output results/candidate.json
ragstat compare --baseline benchmarks/baseline.json \
  --candidate results/candidate.json --thresholds configs/thresholds.yaml \
  --output results/comparison.json
```

## Threshold semantics

```yaml
max_drop:
  mrr: 0.02
  recall@1: 0.02
  answer_token_f1: 0.03
max_increase_ratio:
  estimated_cost_usd: 0.10
```

Quality tolerances are **absolute** differences: 0.90 to 0.88 is a drop of 0.02.
Resource tolerances are **relative** increases: 0.10 permits 10%. Equality passes
(with a small floating-point tolerance); improvements also pass. A zero resource
baseline permits only a zero candidate. Missing measurements fail validation;
omit optional metrics when those stages are disabled. Empty threshold sets and
unknown names are rejected.

Supported quality names: `mrr`, `map`, `recall@K`, `ndcg@K`, and `precision@K`
for measured positive K, `answer_exact_match`, `answer_token_f1`,
`context_token_precision`, `judge_correctness`, `judge_faithfulness`. The three
answer names are lexical overlap. `context_token_precision` does not establish
factual support. Resource names: `retrieval_p95_ms`, `generation_p95_ms`,
`estimated_cost_usd`.

## Paired statistics

```yaml
statistics:
  seed: 0
  bootstrap_samples: 10000
  permutation_samples: 10000
  confidence: 0.95
  gate_on_ci: false
```

When `statistics` is set, each quality threshold also records the mean paired
delta `(candidate - baseline)`, a percentile bootstrap interval for that mean,
and a two-sided sign-flip permutation p-value. Seeds are the strings
`ragbench-stats-v1:{seed}:bootstrap:{metric}` and
`ragbench-stats-v1:{seed}:permutation:{metric}`, drawn with `random.Random`.
The seed prefix `ragbench-stats-v1`, the rubric id `ragbench-judge-v1`, and the
judge tool name `ragbench_judge` keep the original name so saved reports stay valid.
Reports committed before the rename keep `"versions": {"ragbench": ...}` for
compatibility, alongside those legacy schema IDs.
The interval is uncertainty from resampling this fixed question list. It is not
a model of retrieval noise or hardware. The permutation p-value is
`(extreme + 1) / (samples + 1)` and cannot be zero.

`gate_mode` chooses how that interval, or the point estimate, is turned into
pass or fail. `max_drop` is the allowed negative margin in every mode.
Resource thresholds stay relative and are not bootstrapped.

| `gate_mode` | Decision |
| --- | --- |
| `point_drop` | Fail when the point estimate drops by more than `max_drop`. |
| `proven_regression` | Fail only when the upper confidence bound of `(candidate - baseline)` is below `-max_drop`. A wide interval keeps a real drop from failing. |
| `non_inferior` | Fail unless the lower confidence bound of `(candidate - baseline)` is at least `-max_drop`. A wide interval fails until the candidate is shown to stay inside the margin. |

`gate_on_ci: false` leaves the mode at `point_drop`. `gate_on_ci: true` is the
older spelling of `proven_regression` and conflicts with any other `gate_mode`.
`configs/thresholds.yaml` and `configs/scifact-thresholds.yaml` both set
`gate_on_ci: false`, so CI uses point-drop thresholds. The support fixture
allows a 0.02 drop in MRR, Recall@1, and Recall@3.

The support fixture has 27 questions. One changed question moves the mean a
little and leaves an interval wide enough that `proven_regression` can pass
while `non_inferior` fails. A 300-query benchmark narrows the same kind of
interval. The two interval policies answer different questions; CI stays on
`point_drop` because that rule does not depend on the interval width.

## Failing gate

```bash
uv run python scripts/demonstrate_gate_failure.py results/degraded-baseline.json
uv run ragstat compare --baseline benchmarks/baseline.json \
  --candidate results/degraded-baseline.json \
  --thresholds configs/thresholds.yaml
```

The first command lowers MRR, Recall@1, and Recall@3 by 0.07, prints both
metric tables, and exits 0 because the gate rejected the candidate. The second
command is `ragstat compare` itself and exits 2. In GitHub Actions the job
`gate-negative-control` runs the first command and appends the tables to the
job summary. That job is green when the rejection happens. It is a labeled
negative control, separate from the happy-path support-fixture and SciFact
gates.

Every gated quality metric also lists per-question changes, sorted by metric
then question ID, with direction `improved`, `regressed`, or `unchanged`
(absolute tolerance `1e-12`).

Exit codes: **0** passes; **2** means a measured regression; **1** means invalid
inputs, missing metrics, incompatible reports, or another execution error.
CI must treat both nonzero codes as failures.

## Compatibility checks

Reports must have matching schema, corpus and benchmark fingerprints, question
count, and question IDs. This prevents comparing results over different data.
Configurations may differ: that is the purpose of an experiment. Judge gates
additionally require identical judge settings, resolved model identity, and rubric
version. Latency gates require matching recorded environment metadata. Matching
metadata does not control thermal state, background load, or hardware contention;
use a controlled runner and repeated measurements for performance decisions.
Default CI gates only retrieval quality.

Report schema 2 includes timings and optional answer stages. Re-run evaluation to
replace schema 1 reports; comparison does not infer missing measurements.

## Updating the baseline

1. Review dataset or configuration changes and per-question ranking differences.
2. Run the accepted configuration and examine its quality and resource costs.
3. Save the reviewed report as `benchmarks/baseline.json` in a separate change.
4. Explain why the new behavior is accepted; never update merely to turn CI green.

The committed fixture report uses repository-relative paths for portability.
Normal CLI output records resolved local paths. Scrub local paths before publishing
reports from private corpora. Reports also include questions and generated text.

## CI contract

`.github/workflows/ci.yml` installs from `uv.lock` with uv, then runs Ruff,
mypy, tests with a coverage floor, the support-fixture gate, and a SciFact BM25
gate on Python 3.12/Linux and macOS. Those gates fail the workflow on exit 2.
The separate `gate-negative-control` job degrades the support fixture by 0.07
and stays green only when `point_drop` rejects that candidate. Its metric table
is written to the Actions job summary. Separate Linux jobs run real-model
retrieval and Docker. Candidate/comparison JSON is uploaded as a workflow
artifact, including on failure. No credentials or paid provider calls are
needed. The SciFact zip is downloaded in CI and rejected unless its SHA-256
matches.
