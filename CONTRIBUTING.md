# Contributing

Use Python 3.12 and install `.[dev,api,llm,dense]` in a virtual environment.

Before proposing a change, run:

```bash
ruff check .
ruff format --check .
python -m pytest -q
ragbench evaluate --config configs/baseline.yaml --output results/candidate.json
ragbench compare --baseline benchmarks/baseline.json \
  --candidate results/candidate.json --thresholds configs/thresholds.yaml
```

Changes to the real embedding adapter also need `pytest --run-integration`; this
may download the pinned public model. Provider tests must use mock transports,
never credentials or paid API calls. Explain the behavior change and relevant
validation in the pull request. Keep core retrieval logic independent of provider,
HTTP, and storage concerns. Use type hints and add tests for meaningful contracts.

Do not commit secrets, private corpora, generated local reports, or local provider
configuration. Review baseline changes explicitly; a failing gate is a signal to
investigate, not a reason to overwrite its expected result.
