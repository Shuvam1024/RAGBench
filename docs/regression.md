# Regression testing

A comparison answers whether a candidate's measured behavior stays within the
allowed change from a baseline. It never silently changes that baseline.

```bash
ragbench evaluate --config configs/baseline.yaml --output results/candidate.json
ragbench compare --baseline benchmarks/baseline.json \
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
`context_token_precision`, `judge_correctness`, `judge_faithfulness`. Resource
names: `retrieval_p95_ms`, `generation_p95_ms`, `estimated_cost_usd`.

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
The interval is uncertainty from resampling this fixed question list. It is not
a model of retrieval noise or hardware. The permutation p-value is
`(extreme + 1) / (samples + 1)` and cannot be zero.

`gate_on_ci: false` keeps the point-drop rule. `gate_on_ci: true` replaces it:
the check fails only when the interval's upper bound is below `-tolerance`.
That is a more conservative failure rule than the point estimate. Resource
thresholds stay relative and are not bootstrapped.

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
gate on Python 3.12/Linux and macOS. Separate Linux jobs run real-model retrieval
and Docker. Candidate/comparison JSON is uploaded as a workflow artifact,
including on failure. No credentials or paid provider calls are needed. The
SciFact zip is downloaded in CI and rejected unless its SHA-256 matches.
