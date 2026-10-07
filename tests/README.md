# Testing RAGBench

The suite separates offline correctness checks from two explicit real-model
integration tests. Provider tests use mocked HTTP responses and make no paid calls.

```bash
python -m pip install -e ".[dev,api,llm,dense]"
uv run pytest -q --cov=ragbench --cov-fail-under=85
uv run pytest -q --run-integration
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
  negative scores, invalid K, empty input, duplicate IDs, index replacement,
  and the stem tokenizer's stopword removal.
- `test_trec_agreement.py`: Recall, Precision, nDCG, MAP, and MRR on a fixed
  fixture against `pytrec_eval`.
- `test_metrics.py`: hand-calculated Recall@K, MRR, Precision@K, average
  precision, and nDCG, duplicate hits, multiple relevant documents, no matches,
  and invalid metric arguments.
- `test_statistics.py`: seeded bootstrap intervals and sign-flip p-values.
- `test_dataset.py`: SciFact and NFCorpus zip checksums, path safety, JSONL
  ordering, grade handling on a synthetic archive, and train/test benchmark
  filenames. NFCorpus is a confirmation corpus, not a tuning split.
- `test_train_selection.py`: train-split tie breaks for chunk size, tokenizer,
  and hybrid dense weight.
- `test_readme_figures.py`: README metric tables match the committed reports
  at four decimals.
- `test_sweep.py`: a one-factor sweep changes only the named axis and skips the
  base value.
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

- `test_comparison.py`: tolerance boundaries, incompatible inputs, missing metrics,
  zero cost baselines, per-question changes, the confidence-interval gate, and
  CLI pass/regression/error exit codes.
- `test_generation.py`: prompt payloads, usage/cost arithmetic, strict judge JSON,
  malformed/incomplete responses, timeout/error redaction, lexical metrics,
  aggregate consistency, judge identity gates, and local answer evaluation.
- `test_storage_api.py`: transactional inserts, duplicates, read-only missing-file
  behavior, parameterized queries, API responses, and CLI output collisions.

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
