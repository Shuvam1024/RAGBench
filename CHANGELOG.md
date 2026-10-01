# Changelog

All notable changes to RAGBench are documented here.

## [0.1.0] - 2026-10-01

### Added

- Python 3.12 package with a Typer evaluation command and optional JSON reports.
- Strict Pydantic configuration and benchmark validation with portable YAML paths.
- UTF-8 text/Markdown loading, stable document IDs, and deterministic overlapping chunks.
- BM25, sentence-transformers/FAISS dense retrieval, and normalized weighted hybrid retrieval.
- Document-level Recall@K and full-ranking Mean Reciprocal Rank.
- Per-question rankings, input fingerprints, dependency versions, and model provenance.
- Three example configurations, a teaching corpus, and a tested dependency snapshot.
- 82 offline tests and two real-model CLI integration tests.
- Architecture, setup, benchmark, learning, and portfolio documentation.

### Platform notes

- PyTorch inference uses one CPU thread. Real-model integration tests run CLI
  commands in fresh processes to avoid native runtime conflicts on macOS.

### Scope

- Quality regression gates, generation, answer/faithfulness scoring, latency and
  cost analysis, database history, API/UI, Docker, and CI remain future milestones.

[0.1.0]: https://github.com/Shuvam1024/RAGBench
