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
uv sync --frozen --python 3.12 --extra dev --extra api
uv run ragbench evaluate --config configs/baseline.yaml --output results/candidate.json
uv run ragbench compare --baseline benchmarks/baseline.json \
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
Each of these 12 documents is shorter than the default 120-word window, so that
run never emits a second chunk.

## SciFact

`ragbench dataset scifact` downloads the BEIR SciFact zip and rejects it unless
its SHA-256 is `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
The committed length manifest is [benchmarks/scifact/corpus_stats.json](benchmarks/scifact/corpus_stats.json):
5,183 documents, 300 test queries, 339 qrels, every qrel grade 1. Word length
(`\S+` spans) has minimum 33, median 204, mean 214.6280146633224, and maximum
1,541. 4,738 documents are longer than 120 words, so the default chunker splits
them.

BM25 (`k1=1.5`, `b=0.75`, chunk size 120, overlap 20) from
[benchmarks/scifact/bm25.json](benchmarks/scifact/bm25.json):

| Metric | Value |
| --- | --- |
| Documents / chunks / multi-chunk documents | 5183 / 12647 / 4738 |
| Recall@10 | 0.7706666666666666 |
| Recall@100 | 0.8728888888888889 |
| nDCG@10 | 0.6433126633988724 |
| Precision@10 | 0.08433333333333333 |
| MAP | 0.6034257029278666 |
| MRR (full ranking) | 0.6136324295824112 |

The same chunking with `all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, one thread, from
[benchmarks/scifact/dense.json](benchmarks/scifact/dense.json). 174 chunk texts
exceeded the 256-token limit and were truncated:

| Metric | BM25 | Dense MiniLM | Hybrid min-max | Hybrid RRF |
| --- | --- | --- | --- | --- |
| Recall@10 | 0.7706666666666666 | 0.8243333333333334 | 0.8337777777777777 | 0.8171111111111111 |
| Recall@100 | 0.8728888888888889 | 0.9426666666666667 | 0.9443333333333334 | 0.9610000000000001 |
| nDCG@10 | 0.6433126633988724 | 0.6698756028274417 | 0.7086771422459537 | 0.6978116364934386 |
| Precision@10 | 0.08433333333333333 | 0.09233333333333334 | 0.09266666666666667 | 0.091 |
| MAP | 0.6034257029278666 | 0.621310775891958 | 0.6699003860776047 | 0.661227195495906 |
| MRR | 0.6136324295824112 | 0.631546372294966 | 0.6763456448001468 | 0.6736098317639135 |

Sources: [bm25.json](benchmarks/scifact/bm25.json),
[dense.json](benchmarks/scifact/dense.json),
[hybrid-minmax.json](benchmarks/scifact/hybrid-minmax.json)
(`fusion: minmax`, `dense_weight: 0.5`), and
[hybrid-rrf.json](benchmarks/scifact/hybrid-rrf.json) (`fusion: rrf`, `rrf_k: 60`).
Among these four reports, min-max hybrid at `dense_weight` 0.5 has the highest
nDCG@10, MAP, and MRR, and RRF has the highest Recall@100. The weight sweep
below is higher still at 0.75. Precision@10 stays near 0.09 because most
questions have one relevant document, so the maximum Precision@10 is 0.1.

### BM25 ablation

[benchmarks/scifact/ablation-bm25.json](benchmarks/scifact/ablation-bm25.json)
changes one field at a time around the BM25 configuration above. The `base` row
matches that report. nDCG@10 is highest at chunk size 240
(0.652031328715394) and lowest at chunk size 60 (0.610446240944331). MAP is
highest at chunk size 240 (0.6160324275056419). Among `k1` values, 1.5 has the
highest MAP (0.6034257029278666). Setting `b` to 0 raises MAP to
0.6093889989164758 and lowers Recall@10 to 0.7653888888888889.

| Setting | Chunks | Recall@10 | nDCG@10 | MAP |
| --- | --- | --- | --- | --- |
| base (120 / overlap 20 / k1 1.5 / b 0.75) | 12647 | 0.7706666666666666 | 0.6433126633988724 | 0.6034257029278666 |
| chunk_size 60 | 27747 | 0.7362222222222222 | 0.610446240944331 | 0.5725722091625471 |
| chunk_size 240 | 6984 | 0.7673333333333333 | 0.652031328715394 | 0.6160324275056419 |
| chunk_size 480 | 5236 | 0.7739999999999999 | 0.6510804364000519 | 0.6124775495112417 |
| overlap 0 | 11987 | 0.7656666666666666 | 0.6390606916394254 | 0.5993497782270127 |
| overlap 40 | 13856 | 0.7606666666666666 | 0.635057773043908 | 0.5962331871688584 |
| overlap 80 | 20051 | 0.7687222222222222 | 0.638849900534349 | 0.5985841758505063 |
| k1 0.5 | 12647 | 0.7595555555555555 | 0.6400868419204767 | 0.6028959394095554 |
| k1 1.2 | 12647 | 0.7662222222222222 | 0.6398968575806755 | 0.6003470843330199 |
| k1 2.0 | 12647 | 0.7673333333333333 | 0.6354825181089429 | 0.5944920278772259 |
| b 0.0 | 12647 | 0.7653888888888889 | 0.6465302699817467 | 0.6093889989164758 |
| b 0.3 | 12647 | 0.7623333333333333 | 0.6423073276044944 | 0.6051752086987539 |
| b 1.0 | 12647 | 0.7685555555555555 | 0.6367394989174108 | 0.595717983951195 |

### Hybrid ablation

[benchmarks/scifact/ablation-hybrid.json](benchmarks/scifact/ablation-hybrid.json)
varies `dense_weight` and `fusion` around the min-max hybrid configuration.
`dense_weight` 0 matches the BM25 report and `dense_weight` 1 matches the dense
report on every metric above. The highest nDCG@10, MAP, and MRR in the sweep
are at `dense_weight` 0.75. RRF (`rrf_k` 60) matches
[hybrid-rrf.json](benchmarks/scifact/hybrid-rrf.json).

| Setting | Recall@10 | nDCG@10 | MAP | MRR |
| --- | --- | --- | --- | --- |
| dense_weight 0 (BM25) | 0.7706666666666666 | 0.6433126633988724 | 0.6034257029278666 | 0.6136324295824112 |
| dense_weight 0.25 | 0.8085 | 0.6789228904880594 | 0.6368858684859617 | 0.6469993777795833 |
| dense_weight 0.5 | 0.8337777777777777 | 0.7086771422459537 | 0.6699003860776047 | 0.6763456448001468 |
| dense_weight 0.75 | 0.8371111111111111 | 0.7252294354215534 | 0.6899922087149296 | 0.7006330362736721 |
| dense_weight 1 (dense) | 0.8243333333333334 | 0.6698756028274417 | 0.621310775891958 | 0.631546372294966 |
| fusion rrf | 0.8171111111111111 | 0.6978116364934386 | 0.661227195495906 | 0.6736098317639135 |

The full report for `dense_weight` 0.75 is
[hybrid-w075.json](benchmarks/scifact/hybrid-w075.json). It matches the sweep
row. [bm25-vs-hybrid-w075.json](benchmarks/scifact/bm25-vs-hybrid-w075.json)
compares it with the BM25 report under
[configs/scifact-thresholds.yaml](configs/scifact-thresholds.yaml): seed 0,
10,000 bootstrap samples, 10,000 sign-flips, 95% percentile interval,
`gate_on_ci` false. Positive delta means the hybrid score is higher. The
permutation p-value is 0.00009999000099990002 on every metric below, which is
`1 / (10000 + 1)` when every sign-flip was less extreme than the observed mean.
Question counts are improved / regressed / unchanged.

| Metric | Mean delta | 95% CI low | 95% CI high | Improved | Regressed | Unchanged |
| --- | --- | --- | --- | --- | --- | --- |
| MRR | 0.08700060669126096 | 0.05489705624961859 | 0.1196052776794214 | 109 | 36 | 155 |
| MAP | 0.08656650578706318 | 0.05489012743235754 | 0.1190051064464023 | 119 | 37 | 144 |
| Recall@10 | 0.06644444444444444 | 0.03477500000000002 | 0.10033611111111108 | 30 | 5 | 265 |
| Recall@100 | 0.08377777777777777 | 0.05211111111111111 | 0.1176694444444444 | 31 | 2 | 267 |
| nDCG@10 | 0.08191677202268108 | 0.053070846973144986 | 0.11094302901865047 | 88 | 26 | 186 |

## Judge sample

[benchmarks/support/judge-sample.json](benchmarks/support/judge-sample.json)
judges two questions copied from the support fixture (`api-keys-1`,
`rate-limits-1`). Answers come from the extractive generator. The judge is
`gpt-5-nano`, and the response records `gpt-5-nano-2025-08-07`. Rates in
[configs/support-judge.yaml](configs/support-judge.yaml) are the published
text prices of $0.05 input, $0.005 cached input, and $0.40 output per million
tokens. The report's own totals are 1,236 input tokens, 2,692 output tokens,
estimated cost $0.0011386000000000002, mean correctness 0.5, and mean
faithfulness 0.625 (scores divided by 4). One answer was the Markdown heading
rather than the expiration sentence, and the judge scored that correctness 0.
There are no human labels and no agreement statistic.

## Capabilities

- **Retrieval:** BM25, exact dense cosine search, min-max weighted hybrid, and
  reciprocal rank fusion; deterministic chunks, IDs, and ranking ties.
- **Evaluation:** document Recall@K, nDCG@K, Precision@K, MAP, and full-ranking
  MRR; optional normalized exact match, token F1, and context token overlap.
- **Optional LLM judge:** correctness and faithfulness scored against an explicit,
  versioned rubric. Invalid or incomplete responses fail the run.
- **Operational metrics:** indexing and query latency, generation/judge timings,
  reported token usage, and cost estimates using supplied pricing.
- **Regression gates:** absolute quality-drop and relative resource-increase
  thresholds, optional paired bootstrap intervals and sign-flip tests, per-question
  change lists, input compatibility checks, and meaningful CLI exit codes.
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
uv sync --frozen --python 3.12 --extra dev --extra api --extra dense --extra llm
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=ragbench --cov-report=term-missing --cov-fail-under=85
uv run pytest -q --run-integration
```

Offline tests cover retrieval mathematics, report validation, regression failures,
provider HTTP contracts, judge parsing, storage, and API behavior. Two additional
integration tests use real MiniLM embeddings through the dense and hybrid CLIs.
[benchmarks/pytest-junit.xml](benchmarks/pytest-junit.xml) records 135 tests,
0 failures, and 2 skipped. [benchmarks/test-coverage.json](benchmarks/test-coverage.json)
records 1,728 statements, 1,540 covered, 188 missing, and `percent_covered`
89.12037037037037. CI fails the job under 85% and also runs mypy. It enforces
the support-fixture baseline and a SciFact BM25 gate on Linux and macOS, runs
real-model integration checks on Linux, and builds/runs Docker from `uv.lock`.
It makes no paid LLM requests. See the [testing guide](tests/README.md).

## Architecture

```text
text / Markdown → stable documents → word chunks → BM25 / dense / hybrid
                                                        ↓
                                                document rankings
                                                        ↓
                          Recall@K + nDCG + P@K + MAP + MRR + timings
                                                        ↓
                       optional generation → answer metrics → optional judge
                                                        ↓
                                  JSON / SQLite → comparison gates / local API
```

The evaluator supplies question text to retrieval and retrieved context to
generation. Reference answers are reserved for evaluation. Core modules remain
independently testable; provider, storage, and HTTP layers are optional.

## How this compares

RAGBench is a small, typed retrieval-evaluation library. Neighboring tools
optimize different contracts:

- **Ragas** scores generation and retrieval with LLM-judged metrics on each
  sample. RAGBench's primary metrics are deterministic functions of a labeled
  ranking. The optional judge is a separate stage with a versioned rubric,
  strict JSON validation, and caller-supplied token prices.
- **DeepEval** packages LLM metrics as pytest-style checks and can report to a
  hosted product. RAGBench keeps the gate local: fingerprints, thresholds, and
  a failing process exit, with a read-only API that cannot start a paid run.
- **promptfoo** is a prompt and assertion matrix, including red-team cases.
  RAGBench does not vary prompts. It varies retrievers, chunking, and fusion,
  and it refuses to compare reports whose corpus or question fingerprints differ.
- **ARES** trains or prompts a judge to predict human preference and puts
  intervals around that prediction. RAGBench's intervals resample a fixed labeled
  query set. They are not a substitute for human agreement, and this repository
  does not synthesize relevance labels.

## Design limits

Indexes are rebuilt in memory and searched over the full corpus. That keeps
ranking and fusion inspectable. It is the wrong shape for a multi-million
document production index.

The committed BM25 baseline on the 12-document fixture has Recall@3 and
Recall@5 of 1.0, and its document count equals its chunk count, so that
configuration never splits a document. SciFact is the public measurement. It is one BEIR
dataset, not the BEIR suite. FiQA and NFCorpus are not run here; a much larger
passage corpus would multiply CPU embedding time without changing the method.
Reported SciFact scores use whitespace word windows and then collapse chunk
hits to documents. They are not a reproduction of the official BEIR leaderboard,
which scores the dataset's own passages.

SciFact test qrels in this export are all grade 1, so nDCG is binary on that
run even though the metric accepts grades. Precision@K uses K as the denominator,
so Precision@10 stays near 0.1 when a question has a single relevant document.
MAP and MRR use the full document ranking. The JSON report stores only
`stored_hits` documents per question.

The bootstrap interval is uncertainty from resampling this fixed query list.
It is not retrieval noise, corpus sampling, or hardware noise. The permutation
test is a two-sided sign flip of the mean paired delta. `gate_on_ci` is off
unless a threshold file turns it on, and it replaces the point-drop rule rather
than adding a second one. Latencies are one pass, not a load test.

MiniLM runs on CPU at one PyTorch thread with the revision pinned in the dense
configs. A word window can still exceed the 256-token limit; truncation is
warned and counted. The support-fixture judge sample is one live model
assessment of extractive answers. It has no human labels and no agreement
statistic. API cost excludes local compute. The local API has no authentication.

## Documentation

- [Setup and commands](docs/getting-started.md)
- [Architecture and tradeoffs](docs/architecture.md)
- [Benchmark format and metrics](docs/benchmark-format.md)
- [Regression testing](docs/regression.md)
- [Generation, judging, and cost accounting](docs/providers.md)
- [Testing](tests/README.md) · [Contributing](CONTRIBUTING.md)
- [MIT license](LICENSE)
