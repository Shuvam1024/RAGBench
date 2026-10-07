# Contributing

Use Python 3.12 and install `.[dev,api,llm,dense]` in a virtual environment.

Before proposing a change, run:

```bash
uv sync --frozen --python 3.12 --extra dev --extra api
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=ragstat --cov-report=term-missing --cov-fail-under=85
uv run ragstat evaluate --config configs/baseline.yaml --output results/candidate.json
uv run ragstat compare --baseline benchmarks/baseline.json \
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
