UV ?= uv

.PHONY: check lint test dataset-scifact eval-scifact sweep-scifact

check: lint test

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

test:
	$(UV) run pytest -q --cov=ragstat --cov-report=term-missing --cov-fail-under=85

dataset-scifact:
	$(UV) run ragstat dataset scifact --cache .cache/beir --manifest benchmarks/scifact/corpus_stats.json
	$(UV) run ragstat dataset scifact --split train --cache .cache/beir --manifest benchmarks/scifact/corpus_stats_train.json

# Original whitespace chunked reports. This does not run the frozen configs.
eval-scifact: dataset-scifact
	$(UV) run ragstat evaluate --config configs/scifact-bm25.yaml --output benchmarks/scifact/bm25.json --portable
	$(UV) run ragstat evaluate --config configs/scifact-hybrid.yaml --output benchmarks/scifact/hybrid.json --portable

sweep-scifact: dataset-scifact
	$(UV) run ragstat sweep --config configs/scifact-hybrid-train-sweep.yaml --output benchmarks/scifact/hybrid-train-sweep.json
