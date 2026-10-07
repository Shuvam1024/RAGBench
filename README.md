# ragstat

[![CI](https://github.com/Shuvam1024/ragstat/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuvam1024/ragstat/actions/workflows/ci.yml)

**Benchmark RAG configurations and catch quality regressions before shipping.**

RAG (retrieval-augmented generation) answers a question from retrieved text.
ragstat freezes a retriever on a training split, scores a held-out split once,
and fails the process when a candidate drops below a saved baseline. BM25
(Best Matching 25) is the lexical ranker. nDCG (normalized discounted cumulative
gain) is the ranking metric used to choose and to report. A CI in the results
below is a confidence interval. The badge above is continuous integration, and
that job enforces an 85% coverage floor.

## Results at a glance

<!-- tables:begin at-a-glance -->
- Frozen hybrid (BM25 + MiniLM, chosen on train, test scored once) versus document-level BM25 on the [SciFact test split](docs/results.md#held-out-scifact-test): nDCG@10 0.729 vs 0.674, delta +0.055, 95% CI [+0.037, +0.075].
- The same frozen settings on [NFCorpus](docs/results.md#nfcorpus-confirmation): hybrid nDCG@10 delta +0.032 over document-level BM25, with the CI above zero on all five metrics.
- The selected chunked index is nearly one chunk per document: only 46 of 5,183 SciFact documents (26 of 3,633 on NFCorpus) exceed the 480-word window. A post-hoc ablation, run after the test numbers were reported and not used to select a setting, scores document-level BM25 with parameters 1.50/0.75. On the [SciFact test split](docs/results.md#held-out-scifact-test) its nDCG@10 is 0.6910, the selected chunked index is 0.6907 (they differ on 5 of 300 queries), and document-level BM25 at 0.90/0.40 is 0.674. The gap of +0.017 is that parameter change, 95% CI [+0.005, +0.028]. Chunking adds about nothing on this split. On train the frozen pair tie at 0.6972 vs 0.6957. The same ablation on [NFCorpus](docs/results.md#nfcorpus-confirmation) changes document-level nDCG@10 by +0.004, and that interval includes zero.
- Cross-encoder rerank, not fine-tuned, is a negative result on the [SciFact test split](docs/results.md#held-out-scifact-test): nDCG@10 -0.028, 95% CI [-0.054, -0.001], rerank p95 2.0 s.
- [SciFact dev](docs/results.md#held-out-scifact-dev) claim verdicts: frozen NLI micro sentence F1 0.214 vs 0.015 for the majority baseline. Evidence-only sentence F1, the mean over claims with gold evidence, is 0.193 vs 0.018. The per-claim mean is 0.384 vs 0.012 because a claim with no gold evidence and no predicted evidence scores as a perfect match. Macro-F1 is 0.477 vs 0.195. The accuracy interval includes zero.
<!-- tables:end at-a-glance -->

## SciFact test, held out

<!-- tables:begin headline -->
| Setup | nDCG@10 | Recall@10 |
| --- | --- | --- |
| Document-level BM25 | 0.674 | 0.796 |
| Selected BM25 | 0.691 | 0.834 |
| Hybrid | 0.729 | 0.854 |
| Rerank | 0.702 | 0.840 |
<!-- tables:end headline -->

Document-level BM25 is the pre-specified baseline. Hybrid is BM25 plus MiniLM,
chosen on train. Rerank is an untuned cross-encoder on that hybrid. The full
tables, NFCorpus, and the verdict write-up are in [Results](docs/results.md).

## Quick start

```bash
git clone https://github.com/Shuvam1024/ragstat.git
cd ragstat
uv sync --frozen --python 3.12 --extra dev --extra api
uv run ragstat evaluate --config configs/baseline.yaml --output results/candidate.json
uv run ragstat compare --baseline benchmarks/baseline.json \
  --candidate results/candidate.json --thresholds configs/thresholds.yaml
```

The default BM25 run needs no credentials or model download. On the included
12-document, 27-question synthetic support fixture:

<!-- tables:begin support-fixture -->
```text
Recall@1   0.833
Recall@3   1.000
Recall@5   1.000
MRR        0.938 (full document ranking)
```
<!-- tables:end support-fixture -->

## Reproduce the SciFact numbers

```bash
uv run ragstat dataset scifact --cache .cache/beir
uv run ragstat evaluate --config configs/scifact-bm25-document.yaml \
  --output results/scifact-document.json
uv run ragstat compare --baseline benchmarks/scifact/bm25-document.json \
  --candidate results/scifact-document.json --thresholds configs/scifact-thresholds.yaml
uv run ragstat evaluate --config configs/scifact-bm25-selected.yaml \
  --output results/scifact-selected.json
uv run ragstat compare --baseline benchmarks/scifact/bm25-selected.json \
  --candidate results/scifact-selected.json --thresholds configs/scifact-thresholds.yaml
```

Continuous integration runs those two compares with a zero point-drop. The
hybrid and rerank rows need the `dense` extra and are not part of that gate.
`configs/scifact-bm25.yaml` is an exploratory chunk-120 run, not the gate.

## Capabilities

- **Retrieval:** BM25, exact dense cosine search, min-max hybrid, and reciprocal
  rank fusion. Ranking ties are deterministic.
- **Evaluation:** Recall@K, nDCG@K, Precision@K, MAP (mean average precision),
  and MRR (mean reciprocal rank). Optional lexical answer overlap does not establish factual support.
- **Verdicts:** a frozen NLI (natural language inference) policy labels SciFact
  claims and evidence sentences. BEIR qrels are retrieval grades, not those
  labels. BEIR (Benchmarking IR) supplies SciFact and NFCorpus.
- **Regression gates:** absolute quality drops and relative cost increases,
  with optional paired intervals. Reports are JSON or SQLite.

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

The evaluator sends question text to retrieval and retrieved context to
generation. Reference answers are used only for scoring.

## How this compares

- [Ragas](https://github.com/explodinggradients/ragas) scores samples with
  LLM-judged metrics ([paper](https://arxiv.org/abs/2309.15217)). ragstat's
  primary metrics are functions of a labeled ranking.
- [DeepEval](https://github.com/confident-ai/deepeval) packages LLM metrics as
  checks and can report to a hosted product. ragstat's gate stays local.
- [promptfoo](https://github.com/promptfoo/promptfoo) varies prompts and
  assertions. ragstat varies retrievers, chunking, and fusion.
- [ARES](https://arxiv.org/abs/2311.09476) puts intervals around a judge of
  human preference. ragstat's intervals resample a fixed labeled query set.

## Design limits

Indexes are rebuilt in memory and searched over the full corpus. That is the
wrong shape for a multi-million document production index. SciFact is one BEIR
dataset. NFCorpus only repeats frozen settings. The bootstrap interval resamples
this query list. It is not retrieval noise or hardware noise. Latencies are one
pass. The local API has no authentication.

## Documentation

- [Results and methodology](docs/results.md)
- [Setup and commands](docs/getting-started.md)
- [Architecture and tradeoffs](docs/architecture.md)
- [Benchmark format and metrics](docs/benchmark-format.md)
- [Regression testing](docs/regression.md)
- [Generation, judging, and cost accounting](docs/providers.md)
- [Testing](tests/README.md) · [Contributing](CONTRIBUTING.md)
- [MIT license](LICENSE)
