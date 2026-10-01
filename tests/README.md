# Testing RAGBench

With the dense extra installed, the release has 82 offline tests and two
explicit real-model integration tests. The full run passed all 84 tests.

```bash
python -m pip install -e ".[dev,dense]"
python -m pytest -q
python -m pytest -q --run-integration
```

A bare pytest run skips the two integration cases and never downloads model
weights. `--run-integration` enables them; `-m integration --run-integration`
runs just those cases. Set `HF_HOME` to the cache used by your CLI runs, and
optionally set `HF_HUB_OFFLINE=1` after the model is cached.

## What the tests prove

- `test_config.py`: YAML errors, field/range validation, retrieval selection,
  schema version, recall cutoffs, and paths independent of the working directory.
- `test_ingestion.py`: exact word windows, overlap, stable IDs, content changes,
  preserved text, Unicode, BOM/newline handling, sorted loading, empty files,
  symlink handling, corpus relocation, and invalid input.
- `test_bm25.py`: known keyword rankings, case/punctuation handling, tied and
  negative scores, invalid K, empty input, duplicate IDs, and index replacement.
- `test_metrics.py`: hand-calculated Recall@K and MRR, duplicate hits, multiple
  relevant documents, no matches, and invalid metric arguments.
- `test_benchmark.py`: nonblank fields, distinct labels and IDs, supported
  versions, readable JSON, and references to documents in the corpus.
- `test_runner_cli.py`: repeated runs, document deduplication, metric aggregation,
  input fingerprints, output precedence, report contents, and input protection.
- `test_dense.py`: known cosine scores from real FAISS, deterministic ties,
  index reuse/replacement, dimensions, nonfinite/zero vectors, and stable norms.
- `test_hybrid.py`: manual fusion calculations, constant/negative scores,
  endpoint weights, empty lexical signals, and aligned chunk identities.
- `test_integration.py`: real MiniLM semantic retrieval through dense and hybrid
  CLI commands, including report contents and resolved model revision.

## Why inject embeddings?

A fake embedding provider returns vectors whose correct similarities we can
calculate on paper. It replaces the learned model, while the FAISS index and
normalization run normally. This makes algorithm tests fast and deterministic.
The two integration cases then verify that the real model and CLI also work.

Integration cases run in fresh subprocesses. They capture native process
failures as test failures and match how users invoke the CLI. This also avoids
mixing already-initialized FAISS and PyTorch OpenMP runtimes inside pytest on
macOS. See [platform notes](../docs/getting-started.md#macos-native-library-behavior).

With only `.[dev]` installed, the dense module's tests skip explicitly because
FAISS is optional; the remaining offline tests still run. A release must be
checked with the complete dependencies and real-model tests, as this one was.

Tests establish behavior on the tested environment. They do not establish
retriever accuracy on unseen domains or performance at production scale.
