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

The 300-query test split was used while the original chunk size, tokenizer, and
hybrid weight were still being compared. Those runs stay in this repository as
exploratory results. Revised settings were selected on the 809-query train
split ([ir-datasets `beir/scifact/train`](https://ir-datasets.com/beir.html#beir/scifact/train))
and committed before this test split was scored again. NFCorpus was not used
to choose any setting. README figures below are rounded to 4 decimals. The
JSON reports keep full precision.

### Definitions

Every index uses the same prepared field: the BEIR title and body joined by
one newline when both are non-empty. `rank_bm25` `BM25Okapi` scores that
field. The stem tokenizer splits on `\w+` after casefold, drops the vendored
NLTK English stopword list, and applies the Snowball English stemmer from
`snowballstemmer`. A document's retrieval score is its highest-scoring chunk.
`retrieval_p95_ms` in these reports is the full-ranking pass, which scores
every chunk. It is a different measurement from a top-K pipeline latency.

| Setup | Role | Tokenizer | Units | BM25 | Other |
| --- | --- | --- | --- | --- | --- |
| Original chunked | Exploratory test run and CI baseline | whitespace `\w+` | 120-word windows, overlap 20 | k1 1.5, b 0.75 | `configs/scifact-bm25.yaml` |
| Selected chunked | Chosen on train, then scored once on test | stem | 480-word windows, overlap 20 | k1 1.5, b 0.75 | `configs/scifact-bm25-selected.yaml` |
| Document-level | Pre-specified comparison baseline | stem | one string per document | k1 0.9, b 0.4 | `configs/scifact-bm25-document.yaml` |
| Selected hybrid | Weight chosen on train after the lexical freeze | stem, same windows as selected chunked | 480-word windows, overlap 20 | k1 1.5, b 0.75 | min-max, dense_weight 0.5 |

`chunking.unit: document` does not apply the chunk-size default stored beside
it. Pyserini's BEIR flat run reports SciFact test nDCG@10 0.679
([2CR](https://castorini.github.io/pyserini/2cr/beir.html)). That figure is
Lucene BM25 with Lucene's English analyzer. The document-level row here is
the separate `rank_bm25` configuration above.

The hybrid dense model is `sentence-transformers/all-MiniLM-L6-v2` at
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, one PyTorch thread,
batch size 32. Chunk texts longer than 256 tokens are truncated.

### Train selection

`scripts/select_scifact_train.py` scores whitespace and stem at chunk sizes
60, 120, 240, and 480, with overlap 20, k1 1.5, and b 0.75. The winner
maximizes train nDCG@10. Ties prefer the smaller window, then whitespace.
Document-level BM25 is scored and cannot win.
[bm25-train-selection.json](benchmarks/scifact/bm25-train-selection.json):

| Candidate | Train nDCG@10 | Recall@10 | Recall@100 | MAP | MRR |
| --- | --- | --- | --- | --- | --- |
| Selected chunked (stem, 480) | 0.6972 | 0.8196 | 0.9316 | 0.6591 | 0.6695 |
| Document-level reference | 0.6957 | 0.8091 | 0.9272 | 0.6608 | 0.6709 |

On that lexical winner, `configs/scifact-hybrid-train-sweep.yaml` scores
`dense_weight` 0, 0.25, 0.5, 0.75, and 1. The highest train nDCG@10 wins, and
a tie would prefer the smaller weight.
[hybrid-weight-train.json](benchmarks/scifact/hybrid-weight-train.json):

| dense_weight | Train nDCG@10 |
| --- | --- |
| 0.00 | 0.6972 |
| 0.25 | 0.7145 |
| 0.50 | 0.7331 |
| 0.75 | 0.7325 |
| 1.00 | 0.6628 |

The frozen weight is 0.50.

### Cross-encoder second stage

The candidate generator is that frozen hybrid. It is not retuned here. A
cross-encoder then reorders the first `candidate_k` documents. The model is
`cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, on CPU, with one PyTorch thread
and batch size 32. Its positional limit is 512. `max_length` is 512, so a
query/document pair longer than 512 tokens, including special tokens, is
truncated. The score is the raw model output. This checkpoint's activation is
the identity, so the value is a ranking logit.

Each document is sent as one string: the text of its highest-scoring
first-stage chunk. Other chunks of that document are not joined on and are
not scored separately. `candidate_recall_at_100` is the hybrid's Recall@100
before this reorder. nDCG@10 and MRR below are computed on the reordered
candidate list only. Documents outside that list are not returned.

`retrieval_p95_ms` remains the full-ranking first stage, which scores every
chunk. Min-max fusion normalizes over the whole corpus, so the first stage
does not stop early. `rerank_p95_ms` is the cross-encoder alone.
`pipeline_p95_ms` is the sum, the end-to-end cost of the reranked list.

`scripts/select_scifact_rerank.py` scores candidate depths 20, 50, and 100 on
the SciFact train split only. The highest train nDCG@10 wins. A tie prefers
the smaller depth. The frozen hybrid's train nDCG@10 is 0.7331. The reranked
depths score lower on that metric. Depth 20 is the highest of the three, so
it is the frozen depth. The test split is scored once after that choice is
committed.
[rerank-train-selection.json](benchmarks/scifact/rerank-train-selection.json):

| candidate_k | Train nDCG@10 | Recall@10 | Recall@100 | Candidate Recall@100 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.7211 | 0.8505 | 0.8938 | 0.9551 | 0.6787 | 0.6876 |
| 50 | 0.7166 | 0.8430 | 0.9402 | 0.9551 | 0.6771 | 0.6868 |
| 100 | 0.7116 | 0.8311 | 0.9551 | 0.9551 | 0.6747 | 0.6846 |

Recall@100 on a reranked list is the recall of the returned documents.
Candidate Recall@100 is the first stage, before the reorder. At depth 100
those two recalls match, because reranking only permutes the same 100
documents. At depth 20 the reranked list cannot retrieve a document that the
first stage placed at rank 21–100, so its Recall@100 is lower. The train
report for depth 20 truncated 1,966 of 16,180 query/document pairs. Full-ranking
retrieval p95 was 298.3 ms, rerank p95 was 2,081.4 ms, and the top-K pipeline
p95 was 2,301.7 ms. The pipeline is the full-ranking first stage plus the
cross-encoder. One PyTorch thread.

### Held-out SciFact test

300 queries, 339 binary qrels. Corpus fingerprint `0ae06d7ccabbb805…` and
benchmark fingerprint `cc8042c8f9795069…` match the earlier reports. 5,183
documents. Word length (`\S+`) has minimum 33, median 204, mean 214.628, and
maximum 1,541.

| Setup | Chunks | Recall@10 | Recall@100 | nDCG@10 | P@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original chunked | 12647 | 0.7707 | 0.8729 | 0.6433 | 0.0843 | 0.6034 | 0.6136 |
| Document-level baseline | 5183 | 0.7961 | 0.9260 | 0.6741 | 0.0873 | 0.6368 | 0.6458 |
| Selected chunked | 5236 | 0.8335 | 0.9227 | 0.6907 | 0.0920 | 0.6448 | 0.6555 |
| Selected hybrid | 5236 | 0.8539 | 0.9517 | 0.7294 | 0.0957 | 0.6894 | 0.7003 |

Sources: [bm25.json](benchmarks/scifact/bm25.json),
[bm25-document.json](benchmarks/scifact/bm25-document.json),
[bm25-selected.json](benchmarks/scifact/bm25-selected.json),
[hybrid-selected.json](benchmarks/scifact/hybrid-selected.json).
The selected hybrid truncated 3,699 of 5,236 chunk texts. Full-ranking
retrieval p95 was 28.9 ms for selected chunked BM25, 28.3 ms for
document-level BM25, and 180.6 ms for the hybrid.

Paired deltas use seed 0, 10,000 bootstrap resamples, 10,000 sign-flips, and
a 95% percentile interval (`configs/paired-uncertainty.yaml`). Positive delta
means the candidate is higher. That file sets `max_drop` to 1.0 so the
comparison records the interval. `passed` in those files is a measurement
flag. The CI quality gate remains the zero point-drop check of the original
chunked report in `configs/scifact-thresholds.yaml`.

Selected chunked BM25 minus the document-level baseline
([document-vs-bm25-selected.json](benchmarks/scifact/document-vs-bm25-selected.json)):

| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.0166 | [+0.0049, +0.0283] | 0.0056 |
| Recall@10 | +0.0374 | [+0.0158, +0.0611] | 0.0006 |
| Recall@100 | -0.0033 | [-0.0100, +0.0000] | 1.0000 |
| MAP | +0.0080 | [-0.0039, +0.0201] | 0.2053 |
| MRR | +0.0096 | [-0.0037, +0.0233] | 0.1714 |

nDCG@10 and Recall@10 are higher. The MAP and MRR intervals include zero.
Recall@100 is not higher.

Selected hybrid minus the same document-level baseline
([document-vs-hybrid-selected.json](benchmarks/scifact/document-vs-hybrid-selected.json)):

| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.0553 | [+0.0366, +0.0749] | 0.0001 |
| Recall@10 | +0.0578 | [+0.0308, +0.0869] | 0.0002 |
| Recall@100 | +0.0257 | [+0.0100, +0.0447] | 0.0039 |
| MAP | +0.0526 | [+0.0327, +0.0734] | 0.0001 |
| MRR | +0.0544 | [+0.0335, +0.0762] | 0.0001 |

Selected hybrid minus selected chunked BM25, same windows
([bm25-selected-vs-hybrid.json](benchmarks/scifact/bm25-selected-vs-hybrid.json)):

| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.0387 | [+0.0243, +0.0536] | 0.0001 |
| Recall@10 | +0.0204 | [-0.0004, +0.0425] | 0.0600 |
| Recall@100 | +0.0290 | [+0.0120, +0.0490] | 0.0018 |
| MAP | +0.0446 | [+0.0289, +0.0621] | 0.0001 |
| MRR | +0.0448 | [+0.0282, +0.0627] | 0.0001 |

The Recall@10 interval for that fusion comparison includes zero.

### NFCorpus confirmation

`ragbench dataset nfcorpus` checks SHA-256
`efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`.
The test split has 3,633 documents, 323 queries, and 12,334 qrels (11,758
grade 1 and 576 grade 2). Word length has minimum 17, median 237, mean
233.765, and maximum 1,481. nDCG uses gain `2^grade - 1`, so these nDCG
numbers differ from `pytrec_eval` `ndcg_cut` and from published BEIR nDCG.
The configs copy the frozen SciFact settings.

| Setup | Chunks | Recall@10 | Recall@100 | nDCG@10 | P@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original chunked definition | 9541 | 0.1479 | 0.2407 | 0.2939 | 0.2053 | 0.1480 | 0.4946 |
| Document-level baseline | 3633 | 0.1535 | 0.2560 | 0.3273 | 0.2368 | 0.1624 | 0.5298 |
| Selected chunked | 3667 | 0.1571 | 0.2554 | 0.3326 | 0.2427 | 0.1621 | 0.5411 |
| Selected hybrid | 3667 | 0.1712 | 0.3179 | 0.3595 | 0.2632 | 0.1913 | 0.5670 |

The selected hybrid truncated 2,877 of 3,667 chunk texts. Full-ranking
retrieval p95 was 18.6 ms, 19.4 ms, and 179.7 ms for selected chunked BM25,
document-level BM25, and the hybrid.

Selected chunked BM25 minus document-level BM25
([document-vs-bm25-selected.json](benchmarks/nfcorpus/document-vs-bm25-selected.json)):

| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.0053 | [+0.0004, +0.0099] | 0.0228 |
| Recall@10 | +0.0036 | [-0.0015, +0.0090] | 0.1737 |
| Recall@100 | -0.0007 | [-0.0096, +0.0065] | 0.8926 |
| MAP | -0.0003 | [-0.0034, +0.0023] | 0.8801 |
| MRR | +0.0113 | [-0.0031, +0.0257] | 0.1207 |

On NFCorpus the selected windows are only narrowly higher on nDCG@10. The
other four intervals include zero. That does not confirm a general advantage
for the 480-word stem index over document-level BM25.

Selected hybrid minus document-level BM25
([document-vs-hybrid-selected.json](benchmarks/nfcorpus/document-vs-hybrid-selected.json)):

| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.0322 | [+0.0213, +0.0438] | 0.0001 |
| Recall@10 | +0.0177 | [+0.0076, +0.0295] | 0.0002 |
| Recall@100 | +0.0619 | [+0.0448, +0.0799] | 0.0001 |
| MAP | +0.0289 | [+0.0216, +0.0369] | 0.0001 |
| MRR | +0.0372 | [+0.0176, +0.0566] | 0.0002 |

Those five intervals are above zero. The hybrid comparison is the one that
repeats on this second corpus.
[bm25-selected-vs-hybrid.json](benchmarks/nfcorpus/bm25-selected-vs-hybrid.json)
is the same hybrid minus the selected chunked BM25 index (nDCG@10 delta
+0.0269, CI [+0.0170, +0.0376]).

### Exploratory test-split runs

These looked at the 300 test queries before the freeze. They did not choose
the settings above.

| Setup | Recall@10 | Recall@100 | nDCG@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- |
| Whitespace chunked BM25 | 0.7707 | 0.8729 | 0.6433 | 0.6034 | 0.6136 |
| Stem, 120-word windows | 0.8096 | 0.9116 | 0.6828 | 0.6434 | 0.6539 |
| Dense MiniLM, 120-word windows | 0.8243 | 0.9427 | 0.6699 | 0.6213 | 0.6315 |
| Hybrid min-max, dense_weight 0.5 | 0.8338 | 0.9443 | 0.7087 | 0.6699 | 0.6763 |
| Hybrid RRF, rrf_k 60 | 0.8171 | 0.9610 | 0.6978 | 0.6612 | 0.6736 |
| Hybrid min-max, dense_weight 0.75 | 0.8371 | 0.9567 | 0.7252 | 0.6900 | 0.7006 |

The stem 120-word run is
[bm25-stem-chunk120-exploratory.json](benchmarks/scifact/bm25-stem-chunk120-exploratory.json).
It was scored before the train grid finished. `dense_weight` 0.75 was the
highest nDCG@10 in the test-split hybrid ablation
([ablation-hybrid.json](benchmarks/scifact/ablation-hybrid.json)); the train
sweep later froze 0.50 on the 480-word stem index. One-factor BM25 rows
remain in [ablation-bm25.json](benchmarks/scifact/ablation-bm25.json). The
paired comparison of the exploratory 0.75 hybrid with whitespace BM25 remains
in [bm25-vs-hybrid-w075.json](benchmarks/scifact/bm25-vs-hybrid-w075.json).

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
configuration never splits a document. SciFact is one BEIR dataset. NFCorpus
is a second corpus used only to repeat frozen settings. FiQA is not run.
Reported scores collapse chunk hits to documents. They are not a reproduction
of the official BEIR leaderboard or of Pyserini's Lucene BM25.

SciFact test qrels in this export are all grade 1, so nDCG is binary on that
run even though the metric accepts grades. NFCorpus test qrels include grades
1 and 2. nDCG gain is `2^grade - 1`. `pytrec_eval`'s `ndcg_cut` uses the grade
itself, so the two nDCG numbers match on binary labels and differ when a grade
is greater than 1. MAP, recall, precision, and MRR treat any positive grade as
relevant. Precision@K uses K as the denominator,
so Precision@10 stays near 0.1 when a question has a single relevant document.
MAP and MRR use the full document ranking. The JSON report stores only
`stored_hits` documents per question.

The bootstrap interval is uncertainty from resampling this fixed query list.
It is not retrieval noise, corpus sampling, or hardware noise. The permutation
test is a two-sided sign flip of the mean paired delta. CI thresholds set
`gate_on_ci: false`, which is the `point_drop` policy: a drop larger than
`max_drop` fails. `gate_mode: proven_regression` fails only when the upper
confidence bound of `(candidate - baseline)` is below `-max_drop`.
`gate_mode: non_inferior` fails unless the lower bound stays at or above
`-max_drop`. Latencies are one pass, not a load test.

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
