UV ?= uv

.PHONY: check lint test dataset-scifact eval-scifact sweep-scifact

check: lint test

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

test:
	$(UV) run pytest -q --cov=ragbench --cov-report=term-missing --cov-fail-under=85

dataset-scifact:
	$(UV) run ragbench dataset scifact --cache .cache/beir --manifest benchmarks/scifact/corpus_stats.json
	$(UV) run ragbench dataset scifact --split train --cache .cache/beir --manifest benchmarks/scifact/corpus_stats_train.json

eval-scifact: dataset-scifact
	$(UV) run ragbench evaluate --config configs/scifact-bm25.yaml --output benchmarks/scifact/bm25.json --portable
	$(UV) run ragbench evaluate --config configs/scifact-hybrid.yaml --output benchmarks/scifact/hybrid.json --portable

sweep-scifact: dataset-scifact
	$(UV) run ragbench sweep --config configs/scifact-hybrid-train-sweep.yaml --output benchmarks/scifact/hybrid-train-sweep.json
