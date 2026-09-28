# RAGBench

Evaluate BM25 retrieval on a labeled text/Markdown corpus. Configuration is
validated with Pydantic; deterministic word chunks feed document-level Recall@K
and full-ranking Mean Reciprocal Rank.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ragbench evaluate --config configs/baseline.yaml --output results/baseline.json
python -m pytest -q
```

Reports include per-question document rankings, effective settings, input hashes,
and library versions. Document IDs are relative paths within the documents root.
Indexes are rebuilt in memory for each run. Dense and hybrid retrieval are planned
next. The included corpus is a small fixture for checking pipeline behavior.

Licensed under MIT.
