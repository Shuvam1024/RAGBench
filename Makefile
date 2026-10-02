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

eval-scifact: dataset-scifact
	$(UV) run ragbench evaluate --config configs/scifact-bm25.yaml --output benchmarks/scifact/bm25.json --portable
	$(UV) run ragbench evaluate --config configs/scifact-dense.yaml --output benchmarks/scifact/dense.json --portable
	$(UV) run ragbench evaluate --config configs/scifact-hybrid.yaml --output benchmarks/scifact/hybrid-minmax.json --portable
	$(UV) run ragbench evaluate --config configs/scifact-rrf.yaml --output benchmarks/scifact/hybrid-rrf.json --portable

sweep-scifact: dataset-scifact
	$(UV) run ragbench sweep --config configs/scifact-ablation.yaml --output benchmarks/scifact/ablation-bm25.json
	$(UV) run ragbench sweep --config configs/scifact-hybrid-ablation.yaml --output benchmarks/scifact/ablation-hybrid.json
