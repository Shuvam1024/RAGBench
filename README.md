# RAGBench

Evaluate retrieval configurations for RAG systems with a repeatable local benchmark.
Compare BM25, dense vector search, and weighted hybrid retrieval using document-level
Recall@K and Mean Reciprocal Rank (MRR).

**Version 0.1.0: retrieval evaluation is implemented.** The CLI loads text and
Markdown, creates deterministic chunks, builds an index, evaluates labeled
questions, and prints or exports the results. The pipeline uses Python directly,
without LangChain or LlamaIndex.

## Quick start

Requires Python 3.12. From a terminal:

```bash
git clone https://github.com/Shuvam1024/RAGBench.git
cd RAGBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ragbench evaluate --config configs/baseline.yaml
```

The BM25 baseline runs locally without downloading an embedding model:

```text
RAGBench | bm25
Documents: 3 | Chunks: 6 | Questions: 4
Recall@1   0.7500
Recall@3   1.0000
Recall@5   1.0000
MRR        0.8750 (full document ranking)
```

Save per-question rankings, aggregate metrics, input fingerprints, and settings:

```bash
ragbench evaluate --config configs/baseline.yaml --output results/baseline.json
```

## Dense and hybrid retrieval

```bash
python -m pip install -e ".[dev,dense]"
export HF_HOME="$PWD/.cache/huggingface"
ragbench evaluate --config configs/dense.yaml --output results/dense.json
ragbench evaluate --config configs/hybrid.yaml --output results/hybrid.json
```

The first model run downloads public MiniLM weights. The example configs pin a
model revision; subsequent runs can use `HF_HUB_OFFLINE=1` after the model is cached.
The tested dependency snapshot is in
[requirements/constraints-py312.txt](requirements/constraints-py312.txt).
Use `python -m pip install -c requirements/constraints-py312.txt -e ".[dev,dense]"`
to reproduce those package versions. They were verified on Python 3.12/macOS arm64.

On macOS, use the CLI in a fresh process. PyTorch and FAISS can conflict when their
native OpenMP runtimes are initialized in another order inside an existing Python
session. The real-model adapter uses one PyTorch CPU thread; see the
[platform notes](docs/getting-started.md#macos-native-library-behavior).

## What is measured

- **Recall@K:** the fraction of relevant documents found in the first K unique
  retrieved documents, averaged equally across questions.
- **MRR:** the mean of the reciprocal rank of each question's first relevant
  document, using the full document ranking.

Retrievers score chunks. The evaluator keeps the best-ranked chunk from each
source document before applying document-level metrics. Returning many chunks
from one file cannot fill several document-ranking positions.

The included corpus is a teaching fixture: three documents and four questions.
Dense and hybrid produced Recall@1 of 0.875 and MRR of 1.0 in the verified run.
These scores are demonstrations of the pipeline, not evidence of general
retrieval superiority. Recall@3 and Recall@5 are trivial on a three-document
corpus. See [benchmark definitions](docs/benchmark-format.md).

## Test

```bash
python -m pytest -q
# Explicitly allow real-model checks (downloads weights if absent):
python -m pytest -q --run-integration
```

With the dense extra installed: **82 offline tests pass**, and the complete suite
has **84 passing tests**, including two real-model CLI tests. The default run skips
those two integration tests. The [testing guide](tests/README.md) explains how
synthetic vectors verify real FAISS search without a model download.

## Architecture

```text
YAML settings + benchmark questions
                |
text/Markdown -> documents -> word chunks -> BM25 / dense / hybrid
                                                |
                                       ranked chunk hits
                                                |
                                    unique document ranking
                                                |
                                       Recall@K and MRR
                                                |
                                        terminal / JSON
```

- `ragbench/config.py`: strict Pydantic models and configuration-relative paths.
- `ragbench/ingestion/`: UTF-8 loading, content fingerprints, deterministic windows.
- `ragbench/retrieval/`: shared interface, BM25, sentence-transformers/FAISS, fusion.
- `ragbench/evaluation/`: benchmark/result models, pure metrics, orchestration.
- `ragbench/cli.py`: Typer command, report, and atomic JSON export.
- `configs/`, `datasets/`, `tests/`: reproducible inputs and correctness checks.

Read the [setup guide](docs/getting-started.md),
[architecture decisions](docs/architecture.md),
[implementation record](docs/implementation-plan.md),
[learning guide](docs/learning-guide.md), and
[portfolio notes](docs/portfolio.md).

## Scope and roadmap

This release evaluates retrieval quality. Answer correctness, faithfulness,
latency analysis, token/cost accounting, SQLite history, baseline-versus-candidate
regression gates, FastAPI, UI, Docker, and CI are future milestones. The current
CLI fails on invalid inputs or runtime errors; it does not enforce quality
regression thresholds.

Indexes are rebuilt in memory for each run. Full-corpus scoring keeps the first
implementation understandable and deterministic but is intended for small
corpora. Word windows may exceed an embedding model's subword-token limit;
truncation is warned about and recorded in model metadata.

[MIT license](LICENSE) · [Changelog](CHANGELOG.md)
