# Run and inspect ragstat

ragstat evaluates retrieval and optional answers, checks quality regressions,
and records results as JSON or SQLite. Install from `uv.lock` when you want the
same versions CI uses.

## Environment

From the repository root:

```bash
uv sync --frozen --python 3.12 --extra dev --extra api
uv run ragstat evaluate --config configs/baseline.yaml
```

`uv sync --frozen` installs the versions in `uv.lock`. The default extras do
not build `pytrec_eval`. That package needs a C++ compiler and lives in the
`agreement` extra:

```bash
uv sync --frozen --python 3.12 --extra dev --extra api --extra agreement
```

`pyproject.toml` defines the package and dependency groups. `.python-version`
records Python 3.12 for tools that read it. `.venv`, `.cache`, and generated
`results` are local and excluded from Git. Do not move an existing virtual
environment between directories: its activation script and executables contain
absolute paths. Recreate it after relocating the repository.

The [Python packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
explains editable installs and dependency groups.

## Run all three retrievers

```bash
uv sync --frozen --python 3.12 --extra dev --extra dense
export HF_HOME="$PWD/.cache/huggingface"
uv run ragstat evaluate --config configs/baseline.yaml --output results/baseline.json
uv run ragstat evaluate --config configs/dense.yaml --output results/dense.json
uv run ragstat evaluate --config configs/hybrid.yaml --output results/hybrid.json
```

The dense extra adds sentence-transformers, FAISS, and their numerical
libraries. The first dense or hybrid run downloads MiniLM weights. No paid API
key is required. The example model revision is pinned in YAML and the resolved
revision appears in the report. Once cached, `HF_HUB_OFFLINE=1` prevents Hub
network calls.

CI and Docker install from `uv.lock`:

```bash
uv sync --frozen --python 3.12 --extra dev --extra api
uv run ragstat evaluate --config configs/baseline.yaml
```

The lock resolves CPU torch on non-macOS through the PyTorch CPU index. Dense
extras stay upper-bounded in `pyproject.toml` (`sentence-transformers`, `faiss-cpu`,
`numpy`, `torch`).

## Understand your experiment

`configs/baseline.yaml` chooses the corpus, benchmark, chunk size, overlap,
retriever, and recall cutoffs. Paths in YAML resolve from that YAML file's
folder. The CLI's `--config` and `--output` paths resolve from your terminal's
working directory.

You can copy the baseline YAML, point it at your `.txt`/`.md` documents, and
write questions in the [benchmark format](benchmark-format.md). Each relevance
label must match a path relative to the documents root. Unknown labels fail
before the index or model is loaded.

`output.json_path` is optional. An explicit `--output` takes precedence. Parent
output directories are created automatically, and complete JSON reports replace
older reports atomically. Output inside the corpus or over a source/configuration
file is rejected to avoid changing the next run's inputs.

## Inspect the result

Open a report under `results/`. Begin with one entry in `questions`:

1. Read the query and `relevant_document_ids`.
2. Inspect `retrieved_documents`, ordered from best to worst. Each entry records
   a document ID, its best chunk ID, and that chunk's retrieval score.
3. Calculate Recall@1 and reciprocal rank by hand.
4. Compare your answer with the recorded per-question values.
5. Average the values across questions to reproduce the aggregate metrics.

The report also contains effective configuration, input fingerprints, dependency
versions, model revision, and skipped-empty-file counts. JSON object keys for
Recall@K are strings (`"1"`, `"3"`, `"5"`) because JSON object keys are textual.
When generation is enabled, reference answers are used for lexical metrics and
optional judging. See [providers and scoring](providers.md).

## SciFact

```bash
uv run ragstat dataset scifact --cache .cache/beir --manifest benchmarks/scifact/corpus_stats.json
uv run ragstat dataset scifact --split train --cache .cache/beir
uv run ragstat evaluate --config configs/scifact-bm25-document.yaml --output results/scifact-document.json --portable
uv run ragstat evaluate --config configs/scifact-bm25-selected.yaml --output results/scifact-selected.json --portable
uv run ragstat sweep --config configs/scifact-ablation.yaml --output results/ablation-bm25.json
```

The dataset command downloads the BEIR SciFact zip and rejects it unless the
SHA-256 matches. `--split` defaults to `test` and still writes `benchmark.json`.
`--split train` writes `benchmark.train.json` beside it (809 queries; see
[ir-datasets](https://ir-datasets.com/beir.html#beir/scifact/train)) and does
not replace the test benchmark. `scripts/select_scifact_train.py` chooses
chunk size and the BM25 tokenizer on that train split.
`scripts/lock_dense_weight.py` then locks `dense_weight` from the train hybrid
sweep. Commit those frozen configs before evaluating the test split again.
Earlier test-split runs, including the whitespace ablation and the stem
120-word report, are exploratory and do not choose the frozen settings.

`chunking.unit: document` indexes each prepared document once. That
document-level config is the pre-specified comparison baseline. It is not the
chunked RAG setup and is not a Pyserini reproduction
([Pyserini BEIR 2CR](https://castorini.github.io/pyserini/2cr/beir.html)).
Omitting `tokenizer` keeps the original `\w+` tokenizer. Dense and hybrid
configs need the `dense` extra and a Hugging Face cache.

`ragstat dataset nfcorpus` prepares a second BEIR corpus. Its test qrels
repeat the frozen SciFact settings. Chunk size, tokenizer, and fusion weight
stay on the values committed from the SciFact train split. The held-out
reports are `benchmarks/scifact/bm25-selected.json`,
`benchmarks/scifact/bm25-document.json`, `benchmarks/scifact/hybrid-selected.json`,
and the matching files under `benchmarks/nfcorpus/`. Paired intervals use
`configs/paired-uncertainty.yaml`.

## Tests

```bash
python -m pytest -q
python -m pytest -q --run-integration
```

The first command keeps model downloads disabled and runs the offline suite.
The second also runs semantic-search checks through the dense and hybrid CLIs.
The complete dependency set also exercises storage/API and mocked provider contracts.
See [test coverage](../tests/README.md).

## macOS native library behavior

The initial real-model attempt crashed after loading weights. PyTorch and FAISS
ship native OpenMP runtimes that can conflict on macOS; this is also reported
[upstream](https://github.com/pytorch/pytorch/issues/149201).

The model adapter sets PyTorch inference to one CPU thread. Verified dense and
hybrid CLI runs then completed. A separate test-order issue occurred when FAISS
had already executed before PyTorch was imported into the same test process.
Real-model integration tests therefore invoke the public CLI in fresh subprocesses,
which is also the recommended macOS workflow. Offline FAISS tests still use the
real index in the test process.

This does not claim to repair the underlying native libraries. Mixing these
libraries in an existing notebook or long-lived application may still fail.
No unsafe duplicate-runtime override is set. Future latency comparisons must
record the thread setting; the current report includes it in retriever metadata.

## Regression checks and run history

```bash
ragstat evaluate --config configs/baseline.yaml --output results/candidate.json
ragstat compare --baseline benchmarks/baseline.json \
  --candidate results/candidate.json --thresholds configs/thresholds.yaml

# Deliberate 0.07 drop. The script exits 0 after the gate rejects it.
# The compare command exits 2.
uv run python scripts/demonstrate_gate_failure.py results/degraded-baseline.json
ragstat compare --baseline benchmarks/baseline.json \
  --candidate results/degraded-baseline.json --thresholds configs/thresholds.yaml
ragstat evaluate --config configs/answers.yaml --db results/runs.sqlite
ragstat history --db results/runs.sqlite
ragstat serve --db results/runs.sqlite
```

Open `http://127.0.0.1:8000/docs` for the local API explorer. Use `--limit` with
`history` to bound the listing. The `--db` option overrides `storage.sqlite_path`
in YAML. JSON and SQLite destinations must differ and cannot overwrite inputs.

See [regression rules](regression.md), [optional paid providers](providers.md), and
the [Docker image](architecture.md#12-persistence-and-delivery). Keep custom provider config and API keys
out of Git. The default workflows never require credentials.
