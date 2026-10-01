# Changelog

## [1.0.0] - 2026-10-01

### Added

- Baseline/candidate comparison with validated quality and resource thresholds,
  compatible-input checks, and separate regression/error exit codes.
- Generation interface, deterministic local extraction, and optional direct
  OpenAI Responses adapter with bounded requests and usage reporting.
- Exact match, token F1, context overlap, and an optional versioned correctness
  and faithfulness judge with strict response validation.
- Stage latency, token accounting, and configured API cost estimates.
- SQLite run history and a read-only local FastAPI service.
- Non-root Docker image and GitHub Actions for lint, tests, retrieval regression
  gates, real-model integration, and container execution.
- Synthetic support fixture with 12 documents and 27 questions, plus a committed
  baseline report and documentation for evaluation contracts and limitations.

### Changed

- Reports now use schema version 2. Re-run older experiments before comparing.
- Default examples use the support fixture; `configs/smoke.yaml` retains the
  original small corpus.

## [0.1.0] - 2026-10-01

### Added

- Python 3.12 package, Typer CLI, strict Pydantic configuration, and JSON reports.
- Text/Markdown ingestion, stable document IDs, and deterministic word chunking.
- BM25, sentence-transformers/FAISS dense, and normalized weighted hybrid retrieval.
- Document-level Recall@K, full-ranking MRR, fingerprints, and model provenance.
- Offline retrieval tests and real-model CLI integration tests.
- Architecture, setup, and benchmark documentation.

[1.0.0]: https://github.com/Shuvam1024/RAGBench
[0.1.0]: https://github.com/Shuvam1024/RAGBench
