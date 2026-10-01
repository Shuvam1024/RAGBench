# RAGBench

[![CI](https://github.com/Shuvam1024/RAGBench/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuvam1024/RAGBench/actions/workflows/ci.yml)

**Benchmark RAG configurations and catch quality regressions before shipping.**
RAGBench evaluates retrieval and optional generated answers, compares a candidate
against a saved baseline, and returns a failing exit code when configured quality
thresholds are exceeded.

Built directly with Python 3.12, Pydantic, Typer, sentence-transformers, FAISS,
rank-bm25, SQLite, and FastAPI. No LangChain or LlamaIndex.

## Quick start

```bash
git clone https://github.com/Shuvam1024/RAGBench.git
cd RAGBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,api]'
ragbench evaluate --config configs/baseline.yaml --output results/candidate.json
ragbench compare --baseline benchmarks/baseline.json \
  --candidate results/candidate.json --thresholds configs/thresholds.yaml
```

The default BM25 run needs no credentials or model download. On the included
12-document, 27-question synthetic support fixture:

```text
Recall@1   0.8333
Recall@3   1.0000
Recall@5   1.0000
MRR        0.9383 (full document ranking)
```

This fixture exercises the pipeline; these scores do not establish performance
on unseen domains. The committed [baseline report](benchmarks/baseline.json)
contains per-question evidence, input fingerprints, settings, and versions.

## Capabilities

- **Retrieval:** BM25, exact dense cosine search, and normalized weighted hybrid
  scoring; deterministic chunks, IDs, and ranking ties.
- **Evaluation:** document Recall@K and full-ranking MRR; optional normalized
  exact match, token F1, and context token overlap for answers.
- **Optional LLM judge:** correctness and faithfulness scored against an explicit,
  versioned rubric. Invalid or incomplete responses fail the run.
- **Operational metrics:** indexing and query latency, generation/judge timings,
  reported token usage, and cost estimates using supplied pricing.
- **Regression gates:** absolute quality-drop and relative resource-increase
  thresholds, input compatibility checks, and meaningful CLI exit codes.
- **Results:** atomic JSON reports, SQLite history, a read-only local FastAPI
  service, Docker packaging, and GitHub Actions checks.

## Try another configuration

```bash
# Real embeddings; first run downloads the pinned MiniLM model.
python -m pip install -e '.[dev,dense]'
export HF_HOME="$PWD/.cache/huggingface"
ragbench evaluate --config configs/dense.yaml --output results/dense.json
ragbench evaluate --config configs/hybrid.yaml --output results/hybrid.json

# Exercise answer evaluation locally with deterministic sentence extraction.
ragbench evaluate --config configs/answers.yaml --db results/runs.sqlite
ragbench history --db results/runs.sqlite

# Local read-only API; interactive API documentation at /docs.
python -m pip install -e '.[api]'
ragbench serve --db results/runs.sqlite
```

For opt-in paid generation and judging, see [providers and scoring](docs/providers.md).
On macOS, run dense benchmarks as fresh CLI processes; see the
[native-library notes](docs/getting-started.md#macos-native-library-behavior).

## Docker

```bash
docker build -t ragbench .
docker run --rm ragbench
# Persist reports and history in a named volume.
docker volume create ragbench-data
docker run --rm -v ragbench-data:/data ragbench evaluate \
  --config configs/answers.yaml --output /data/answers.json --db /data/runs.sqlite
docker run --rm -p 127.0.0.1:8000:8000 -v ragbench-data:/data ragbench \
  serve --db /data/runs.sqlite --host 0.0.0.0
```

The image runs as an unprivileged user. Build optional providers with
`--build-arg EXTRAS=api,llm,dense`. The default image includes the API and BM25.

## Tests and CI

```bash
python -m pip install -e '.[dev,api,dense,llm]'
ruff check .
ruff format --check .
python -m pytest -q --cov=ragbench --cov-report=term-missing
python -m pytest -q --run-integration
```

Offline tests cover retrieval mathematics, report validation, regression failures,
provider HTTP contracts, judge parsing, storage, and API behavior. Two additional
integration tests use real MiniLM embeddings through the dense and hybrid CLIs.
CI runs tests on Linux and macOS, enforces the committed quality baseline, runs
real-model integration checks on Linux, and builds/runs Docker. It makes no paid
LLM requests. See the [testing guide](tests/README.md).

## Architecture

```text
text / Markdown → stable documents → word chunks → BM25 / dense / hybrid
                                                        ↓
                                                document rankings
                                                        ↓
                                            Recall@K + MRR + timings
                                                        ↓
                       optional generation → answer metrics → optional judge
                                                        ↓
                                  JSON / SQLite → comparison gates / local API
```

The evaluator supplies question text to retrieval and retrieved context to
generation. Reference answers are reserved for evaluation. Core modules remain
independently testable; provider, storage, and HTTP layers are optional.

## Design limits

Indexes are rebuilt in memory and searched over the full corpus. This keeps
ranking and fusion inspectable but targets small evaluation datasets. Word
windows can exceed embedding token limits; truncation is warned about and recorded.

Lexical overlap is a proxy, and LLM judge outputs are model assessments rather
than verified truth. The judge adapter is tested with controlled HTTP responses;
its quality has not been calibrated with human labels. Latencies are single-run
measurements, not load-test results. API cost excludes local compute and embedding
infrastructure. The local API has no authentication; a dashboard and multi-user
hosting are outside this release.

## Documentation

- [Setup and commands](docs/getting-started.md)
- [Architecture and tradeoffs](docs/architecture.md)
- [Benchmark format and metrics](docs/benchmark-format.md)
- [Regression testing](docs/regression.md)
- [Generation, judging, and cost accounting](docs/providers.md)
- [Testing](tests/README.md) · [Contributing](CONTRIBUTING.md)
- [MIT license](LICENSE)
