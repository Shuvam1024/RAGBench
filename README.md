# ragstat

[![CI](https://github.com/Shuvam1024/ragstat/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuvam1024/ragstat/actions/workflows/ci.yml)

**Benchmark RAG configurations and catch quality regressions before shipping.**
ragstat evaluates retrieval and optional generated answers, compares a candidate
against a saved baseline, and returns a failing exit code when configured quality
thresholds are exceeded.

## Results at a glance

<!-- tables:begin at-a-glance -->
- Frozen hybrid (BM25 + MiniLM, chosen on train, test scored once) versus document-level BM25 on the [SciFact test split](#held-out-scifact-test): nDCG@10 0.729 vs 0.674, delta +0.055, 95% CI [+0.037, +0.075].
- The same frozen settings on [NFCorpus](#nfcorpus-confirmation): hybrid nDCG@10 delta +0.032 over document-level BM25, with the CI above zero on all five metrics.
- Chunked BM25 tuning is a small gain (nDCG@10 +0.017 on the [SciFact test split](#held-out-scifact-test)) that [NFCorpus](#nfcorpus-confirmation) does not confirm.
- Cross-encoder rerank, not fine-tuned, is a negative result on the [SciFact test split](#held-out-scifact-test): nDCG@10 -0.028, 95% CI [-0.054, -0.001], rerank p95 2.0 s.
- [SciFact dev](#held-out-scifact-dev) claim verdicts: frozen NLI macro-F1 0.477 vs 0.195 for the majority baseline. Evidence-sentence F1 is 0.384 vs 0.012. The accuracy interval includes zero.
<!-- tables:end at-a-glance -->

Built directly with Python 3.12, Pydantic, Typer, sentence-transformers, FAISS,
rank-bm25, SQLite, and FastAPI. No LangChain or LlamaIndex.

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
to choose any setting. README metric figures below are rounded to 3 decimals.
Millisecond latencies are whole milliseconds, and latencies in seconds use one
decimal. Sign-flip p-values stay at 4 decimals so the permutation floor does
not display as zero. The JSON reports keep full precision.
`scripts/render_readme_tables.py` rewrites the marked regions from those reports.

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

<!-- tables:begin lexical-train -->
| Candidate | Train nDCG@10 | Recall@10 | Recall@100 | MAP | MRR |
| --- | --- | --- | --- | --- | --- |
| Selected chunked (stem, 480) | 0.697 | 0.820 | 0.932 | 0.659 | 0.670 |
| Document-level reference | 0.696 | 0.809 | 0.927 | 0.661 | 0.671 |
<!-- tables:end lexical-train -->

On that lexical winner, `configs/scifact-hybrid-train-sweep.yaml` scores
`dense_weight` 0, 0.25, 0.5, 0.75, and 1. The highest train nDCG@10 wins, and
a tie would prefer the smaller weight.
[hybrid-weight-train.json](benchmarks/scifact/hybrid-weight-train.json):

<!-- tables:begin hybrid-weights -->
| dense_weight | Train nDCG@10 |
| --- | --- |
| 0.00 | 0.697 |
| 0.25 | 0.714 |
| 0.50 | 0.733 |
| 0.75 | 0.732 |
| 1.00 | 0.663 |

The frozen weight is 0.50.
<!-- tables:end hybrid-weights -->

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
the smaller depth. The frozen hybrid's train nDCG@10 is in the table below.
The reranked depths score lower on that metric. Depth 20 is the highest of the
three, so
it is the frozen depth. The test split is scored once after that choice is
committed.
[rerank-train-selection.json](benchmarks/scifact/rerank-train-selection.json):

<!-- tables:begin rerank-train -->
| candidate_k | Train nDCG@10 | Recall@10 | Recall@100 | Candidate Recall@100 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.721 | 0.850 | 0.894 | 0.955 | 0.679 | 0.688 |
| 50 | 0.717 | 0.843 | 0.940 | 0.955 | 0.677 | 0.687 |
| 100 | 0.712 | 0.831 | 0.955 | 0.955 | 0.675 | 0.685 |

The frozen hybrid's train nDCG@10 is 0.733.
The train report for depth 20 truncated 1,966 of 16,180 query/document pairs. Full-ranking retrieval p95 was 298 ms, rerank p95 was 2,081 ms, and the top-K pipeline p95 was 2,302 ms.
<!-- tables:end rerank-train -->

Recall@100 on a reranked list is the recall of the returned documents.
Candidate Recall@100 is the first stage, before the reorder. At depth 100
those two recalls match, because reranking only permutes the same 100
documents. At depth 20 the reranked list cannot retrieve a document that the
first stage placed at rank 21–100, so its Recall@100 is lower. The pipeline is
the full-ranking first stage plus the cross-encoder. One PyTorch thread.

### Held-out SciFact test

<!-- tables:begin scifact-test -->
Corpus fingerprint `0ae06d7ccabbb805` and benchmark fingerprint `cc8042c8f9795069` are the leading digits of those fields in the original chunked report. 300 queries, 339 binary qrels. 5,183 documents. Word length (`\S+`) has minimum 33, median 204, mean 214.628, and maximum 1,541.

| Setup | Chunks | Recall@10 | Recall@100 | nDCG@10 | P@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original chunked | 12,647 | 0.771 | 0.873 | 0.643 | 0.084 | 0.603 | 0.614 |
| Document-level baseline | 5,183 | 0.796 | 0.926 | 0.674 | 0.087 | 0.637 | 0.646 |
| Selected chunked | 5,236 | 0.834 | 0.923 | 0.691 | 0.092 | 0.645 | 0.655 |
| Selected hybrid | 5,236 | 0.854 | 0.952 | 0.729 | 0.096 | 0.689 | 0.700 |

The selected hybrid truncated 3,699 of 5,236 chunk texts. Full-ranking retrieval p95 was 29 ms for selected chunked BM25, 28 ms for document-level BM25, and 181 ms for the hybrid.
<!-- tables:end scifact-test -->

Sources: [bm25.json](benchmarks/scifact/bm25.json),
[bm25-document.json](benchmarks/scifact/bm25-document.json),
[bm25-selected.json](benchmarks/scifact/bm25-selected.json),
[hybrid-selected.json](benchmarks/scifact/hybrid-selected.json).

<!-- tables:begin paired-settings -->
Paired deltas use seed 0, 10,000 bootstrap resamples, 10,000 sign-flips, and a 95% percentile interval. Retrieval comparison files do not store that recipe. It matches `configs/paired-uncertainty.yaml` and the verdict comparison.
<!-- tables:end paired-settings -->
Positive delta means the candidate is higher. That file sets `max_drop` to 1.0
so the comparison records the interval. `passed` in those files is a measurement
flag. The CI quality gate remains the zero point-drop check of the original
chunked report in `configs/scifact-thresholds.yaml`.

Selected chunked BM25 minus the document-level baseline
([document-vs-bm25-selected.json](benchmarks/scifact/document-vs-bm25-selected.json)):

<!-- tables:begin scifact-doc-vs-selected -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.017 | [+0.005, +0.028] | 0.0056 |
| Recall@10 | +0.037 | [+0.016, +0.061] | 0.0006 |
| Recall@100 | -0.003 | [-0.010, +0.000] | 1.0000 |
| MAP | +0.008 | [-0.004, +0.020] | 0.2053 |
| MRR | +0.010 | [-0.004, +0.023] | 0.1714 |

Above zero: nDCG@10, Recall@10. Intervals that include zero: Recall@100, MAP, MRR.
<!-- tables:end scifact-doc-vs-selected -->

Selected hybrid minus the same document-level baseline
([document-vs-hybrid-selected.json](benchmarks/scifact/document-vs-hybrid-selected.json)):

<!-- tables:begin scifact-doc-vs-hybrid -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.055 | [+0.037, +0.075] | 0.0001 |
| Recall@10 | +0.058 | [+0.031, +0.087] | 0.0002 |
| Recall@100 | +0.026 | [+0.010, +0.045] | 0.0039 |
| MAP | +0.053 | [+0.033, +0.073] | 0.0001 |
| MRR | +0.054 | [+0.033, +0.076] | 0.0001 |

Above zero: nDCG@10, Recall@10, Recall@100, MAP, MRR.
<!-- tables:end scifact-doc-vs-hybrid -->

Selected hybrid minus selected chunked BM25, same windows
([bm25-selected-vs-hybrid.json](benchmarks/scifact/bm25-selected-vs-hybrid.json)):

<!-- tables:begin scifact-selected-vs-hybrid -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.039 | [+0.024, +0.054] | 0.0001 |
| Recall@10 | +0.020 | [-0.000, +0.043] | 0.0600 |
| Recall@100 | +0.029 | [+0.012, +0.049] | 0.0018 |
| MAP | +0.045 | [+0.029, +0.062] | 0.0001 |
| MRR | +0.045 | [+0.028, +0.063] | 0.0001 |

Above zero: nDCG@10, Recall@100, MAP, MRR. Intervals that include zero: Recall@10.
<!-- tables:end scifact-selected-vs-hybrid -->

The frozen cross-encoder reorders the hybrid's first-stage candidates
([rerank-selected.json](benchmarks/scifact/rerank-selected.json)). Recall@100
on the reranked list is lower because the returned list stops at `candidate_k`.

<!-- tables:begin scifact-rerank -->
| Setup | Recall@10 | Recall@100 | nDCG@10 | P@10 | MAP | MRR | Candidate Recall@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reranked top 20 | 0.840 | 0.890 | 0.702 | 0.094 | 0.657 | 0.670 | 0.952 |

The run truncated 767 of 6,000 query/document pairs. Full-ranking retrieval p95 was 248 ms, rerank p95 was 1,995 ms, and the top-K pipeline p95 was 2,168 ms.
<!-- tables:end scifact-rerank -->

Reranked candidates minus the selected hybrid
([hybrid-vs-rerank.json](benchmarks/scifact/hybrid-vs-rerank.json)):

<!-- tables:begin scifact-hybrid-vs-rerank -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | -0.028 | [-0.054, -0.001] | 0.0424 |
| Recall@10 | -0.014 | [-0.041, +0.013] | 0.3180 |
| Recall@100 | -0.062 | [-0.090, -0.037] | 0.0001 |
| MAP | -0.033 | [-0.064, -0.003] | 0.0317 |
| MRR | -0.031 | [-0.062, +0.001] | 0.0619 |

Below zero: nDCG@10, Recall@100, MAP. Intervals that include zero: Recall@10, MRR.
<!-- tables:end scifact-hybrid-vs-rerank -->

The Recall@100 drop is the candidate cutoff. This checkpoint was not
fine-tuned on SciFact.

### NFCorpus confirmation

`ragstat dataset nfcorpus` checks SHA-256
`efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`.
nDCG uses gain `2^grade - 1`, so these nDCG numbers differ from `pytrec_eval`
`ndcg_cut` and from published BEIR nDCG. The configs copy the frozen SciFact
settings.

<!-- tables:begin nfcorpus-test -->
3,633 documents, 323 queries, and 12,334 qrels (11,758 grade 1 and 576 grade 2). Word length has minimum 17, median 237, mean 233.765, and maximum 1,481.

| Setup | Chunks | Recall@10 | Recall@100 | nDCG@10 | P@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original chunked definition | 9,541 | 0.148 | 0.241 | 0.294 | 0.205 | 0.148 | 0.495 |
| Document-level baseline | 3,633 | 0.153 | 0.256 | 0.327 | 0.237 | 0.162 | 0.530 |
| Selected chunked | 3,667 | 0.157 | 0.255 | 0.333 | 0.243 | 0.162 | 0.541 |
| Selected hybrid | 3,667 | 0.171 | 0.318 | 0.359 | 0.263 | 0.191 | 0.567 |

The selected hybrid truncated 2,877 of 3,667 chunk texts. Full-ranking retrieval p95 was 19 ms, 19 ms, and 180 ms for selected chunked BM25, document-level BM25, and the hybrid.
<!-- tables:end nfcorpus-test -->

Selected chunked BM25 minus document-level BM25
([document-vs-bm25-selected.json](benchmarks/nfcorpus/document-vs-bm25-selected.json)):

<!-- tables:begin nfcorpus-doc-vs-selected -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.005 | [+0.000, +0.010] | 0.0228 |
| Recall@10 | +0.004 | [-0.001, +0.009] | 0.1737 |
| Recall@100 | -0.001 | [-0.010, +0.007] | 0.8926 |
| MAP | -0.000 | [-0.003, +0.002] | 0.8801 |
| MRR | +0.011 | [-0.003, +0.026] | 0.1207 |

Above zero: nDCG@10. Intervals that include zero: Recall@10, Recall@100, MAP, MRR. nDCG@10 is above zero, but its lower bound rounds to +0.000.
<!-- tables:end nfcorpus-doc-vs-selected -->

That does not confirm a general advantage for the 480-word stem index over
document-level BM25.

Selected hybrid minus document-level BM25
([document-vs-hybrid-selected.json](benchmarks/nfcorpus/document-vs-hybrid-selected.json)):

<!-- tables:begin nfcorpus-doc-vs-hybrid -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.032 | [+0.021, +0.044] | 0.0001 |
| Recall@10 | +0.018 | [+0.008, +0.029] | 0.0002 |
| Recall@100 | +0.062 | [+0.045, +0.080] | 0.0001 |
| MAP | +0.029 | [+0.022, +0.037] | 0.0001 |
| MRR | +0.037 | [+0.018, +0.057] | 0.0002 |

Above zero: nDCG@10, Recall@10, Recall@100, MAP, MRR.
<!-- tables:end nfcorpus-doc-vs-hybrid -->

The hybrid comparison is the one that repeats on this second corpus.
[bm25-selected-vs-hybrid.json](benchmarks/nfcorpus/bm25-selected-vs-hybrid.json)
is the same hybrid minus the selected chunked BM25 index:

<!-- tables:begin nfcorpus-selected-vs-hybrid -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.027 | [+0.017, +0.038] | 0.0001 |
| Recall@10 | +0.014 | [+0.005, +0.025] | 0.0045 |
| Recall@100 | +0.063 | [+0.047, +0.079] | 0.0001 |
| MAP | +0.029 | [+0.023, +0.036] | 0.0001 |
| MRR | +0.026 | [+0.008, +0.044] | 0.0046 |

Above zero: nDCG@10, Recall@10, Recall@100, MAP, MRR.
<!-- tables:end nfcorpus-selected-vs-hybrid -->

### Exploratory test-split runs

These looked at the 300 test queries before the freeze. They did not choose
the settings above.

<!-- tables:begin exploratory -->
| Setup | Recall@10 | Recall@100 | nDCG@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- |
| Whitespace chunked BM25 | 0.771 | 0.873 | 0.643 | 0.603 | 0.614 |
| Stem, 120-word windows | 0.810 | 0.912 | 0.683 | 0.643 | 0.654 |
| Dense MiniLM, 120-word windows | 0.824 | 0.943 | 0.670 | 0.621 | 0.632 |
| Hybrid min-max, dense_weight 0.50 | 0.834 | 0.944 | 0.709 | 0.670 | 0.676 |
| Hybrid RRF, rrf_k 60 | 0.817 | 0.961 | 0.698 | 0.661 | 0.674 |
| Hybrid min-max, dense_weight 0.75 | 0.837 | 0.957 | 0.725 | 0.690 | 0.701 |
<!-- tables:end exploratory -->

The stem 120-word run is
[bm25-stem-chunk120-exploratory.json](benchmarks/scifact/bm25-stem-chunk120-exploratory.json).
It was scored before the train grid finished. `dense_weight` 0.75 was the
highest nDCG@10 in the test-split hybrid ablation
([ablation-hybrid.json](benchmarks/scifact/ablation-hybrid.json)); the train
sweep later froze 0.50 on the 480-word stem index. One-factor BM25 rows
remain in [ablation-bm25.json](benchmarks/scifact/ablation-bm25.json). The
paired comparison of the exploratory 0.75 hybrid with whitespace BM25 remains
in [bm25-vs-hybrid-w075.json](benchmarks/scifact/bm25-vs-hybrid-w075.json).

## SciFact claim verdicts

`answer_exact_match`, `answer_token_f1`, and `context_token_precision` are
lexical overlap. `context_token_precision` is the fraction of answer tokens
that also appear in the retrieved context. That overlap does not establish factual support.
It is not a faithfulness score. The optional judge
rubric's faithfulness field is a separate 0–4 model score. It is not a SciFact
SUPPORT or CONTRADICT label.

Verdicts and evidence sentence IDs are read from the AllenAI SciFact claim
files, SHA-256
`11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`
([data.tar.gz](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz),
schema in [doc/data.md](https://github.com/allenai/scifact/blob/master/doc/data.md)).
They are not read from BEIR qrels. A qrel row is `query-id`, `corpus-id`, and
an integer `score`. On this export every score is 1. For the train claims and
for the dev claims, that document set equals `cited_doc_ids`. It does not equal
the evidence-document set.

<!-- tables:begin verdict-counts -->
Train has 809 claims: the qrel set matches the evidence documents for 480 of them and differs for the other 329. Dev has 300 claims: 175 match and 125 differ. 1,055 of 5,183 abstracts differ from a single-space join before whitespace is collapsed. Train label counts are SUPPORT 332, CONTRADICT 173, and NEI 304.
<!-- tables:end verdict-counts -->

A claim with no annotated evidence is still cited, and some claims cite a
document that has no rationale. Claim 263 contradicts from documents 11328820
and 30041340 and also cites 14853989, which has no evidence annotation and is
still a grade-1 qrel.

Sentence IDs index the `corpus.jsonl` abstract list. They do not index the
title, and they are not recovered by splitting the BEIR body. After whitespace
is collapsed, the BEIR body matches the abstract sentences joined by spaces.
The mismatch count is in the claim-mapping paragraph above.

The public `claims_test.jsonl` has no evidence field. Those 300 labels are
withheld, so this repository does not score that file and does not treat the
missing field as NEI. Empty evidence on train or dev does mean NEI. No train
or dev claim mixes SUPPORT and CONTRADICT. The gold verdict is that unique
label, or NEI when the evidence object is empty. Gold sentences are the set of
`(document id, abstract sentence index)` pairs.

BEIR query IDs are the SciFact train and dev claim IDs, and the text matches.
BEIR has no separate dev qrels file. The dev claim IDs are the BEIR test query
IDs. The official SciFact test IDs are not in the BEIR zip. The labeled split
scored here is SciFact dev, because it has labels and the public test file
does not. Those dev claims are the same 300 queries the retrieval reports call
the SciFact test split. Verdict knobs are not selected on them.

Retrieval is the frozen document-level BM25 index: stem, k1 0.9, b 0.4, one
string per document. It is not retuned. The indexed string is still the
prepared BEIR title and body.

Predicted sentences are a set. Precision, recall, and F1 are:

- both sets empty: 1, 1, 1, a correct abstention
- predicted empty and gold nonempty: 0, 0, 0
- gold empty and predicted nonempty: 0, 0, 0
- otherwise the overlap divided by the prediction count, the gold count, and
  their harmonic mean

The reported sentence precision, recall, and F1 are means of those per-claim
scores. Micro scores pool the sentence counts across claims. Verdict accuracy
is the fraction of exact label matches. Macro-F1 is the unweighted mean of the
SUPPORT, CONTRADICT, and NEI F1 scores. A class with no gold and no predictions
scores 0.

The baseline predicts the majority train verdict. Those counts are in the
claim-mapping paragraph above, and the majority is SUPPORT. A tie would prefer
NEI, then SUPPORT, then CONTRADICT. The baseline's evidence sentence is
sentence 0 of the top retrieved document. It predicts no sentence when the
verdict is NEI. It does not use the NLI model.

The NLI model is `cross-encoder/nli-MiniLM2-L6-H768` at revision
`b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, on CPU, with one PyTorch thread and
batch size 32. `max_length` is 512. The checkpoint's positional limit is 514.
Pairs longer than 512 tokens, including special tokens, are truncated. The
model was trained on SNLI and MultiNLI. It was not fine-tuned on SciFact. The
premise is the abstract sentence and the hypothesis is the claim. Softmax over
the three logits maps entailment to SUPPORT, contradiction to CONTRADICT, and
neutral to NEI. Equal probabilities break toward contradiction, then
entailment, then neutral. A sentence is kept when that label is not NEI and
its probability is at least `min_confidence`. Kept sentences are ordered by
descending probability, then document ID, then sentence index. The top
`sentence_k` are the prediction. The verdict is the label of the first kept
sentence, which is the highest-ranked label when the kept sentences disagree.
No kept sentence means NEI and an empty evidence set.

`scripts/select_scifact_verdict.py` scores `doc_k` 1 and 3, `sentence_k` 1 and
2, and `min_confidence` 0.5 and 0.7 on the train claims only. The forward pass
covers the top 3 documents once. Smaller document depths reuse those
probabilities. The winner maximizes train macro-F1. Ties prefer a higher mean
sentence F1, then a smaller `doc_k`, then a smaller `sentence_k`, then a
higher `min_confidence`. Accuracy and macro-F1 do not change with `sentence_k`,
because the verdict is the label of the first kept sentence. `sentence_k` only
changes the evidence set. Dev is scored once after that choice is committed.
Tests inject an NLI scorer. The pinned model is not required for those tests.
[verdict-train-selection.json](benchmarks/scifact/verdict-train-selection.json):

<!-- tables:begin verdict-train -->
| doc_k | sentence_k | min_confidence | Accuracy | Macro-F1 | Sentence P | Sentence R | Sentence F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.5 | 0.472 | 0.464 | 0.394 | 0.328 | 0.345 |
| 1 | 1 | 0.7 | 0.497 | 0.475 | 0.426 | 0.371 | 0.385 |
| 1 | 2 | 0.5 | 0.472 | 0.464 | 0.377 | 0.359 | 0.357 |
| 1 | 2 | 0.7 | 0.497 | 0.475 | 0.413 | 0.389 | 0.391 |
| 3 | 1 | 0.5 | 0.422 | 0.421 | 0.281 | 0.229 | 0.243 |
| 3 | 1 | 0.7 | 0.459 | 0.452 | 0.329 | 0.281 | 0.293 |
| 3 | 2 | 0.5 | 0.422 | 0.421 | 0.265 | 0.261 | 0.253 |
| 3 | 2 | 0.7 | 0.459 | 0.452 | 0.310 | 0.297 | 0.295 |

`doc_k` 1, `sentence_k` 2, `min_confidence` 0.7 is the frozen policy. Its train macro-F1 is 0.475 and its mean sentence F1 is 0.391. The majority-SUPPORT baseline on the same train claims has accuracy 0.410, macro-F1 0.194, and mean sentence F1 0.014. The grid scored 23,066 sentence/claim pairs in 840.2 s. The winning depth, scored on its own, was 7,656 pairs in 279.8 s. 0 pairs were truncated. Document-level retrieval p95 was 18 ms.
<!-- tables:end verdict-train -->

The `sentence_k` 1 row at the same `doc_k` and `min_confidence` has the same
train accuracy and macro-F1. The frozen row wins on mean sentence F1. The
`doc_k` 3 rows are lower on train macro-F1. One PyTorch thread.

### Held-out SciFact dev

Scored once after the train freeze. These are the same claim IDs the retrieval
reports call the SciFact test split. The public SciFact test file is still unscored.

<!-- tables:begin verdict-dev -->
300 labeled claims. Gold counts are SUPPORT 124, CONTRADICT 64, and NEI 112.

|  | Accuracy | Macro-F1 | Sentence P | Sentence R | Sentence F1 |
| --- | --- | --- | --- | --- | --- |
| Majority SUPPORT | 0.413 | 0.195 | 0.017 | 0.010 | 0.012 |
| Frozen NLI | 0.493 | 0.477 | 0.410 | 0.382 | 0.384 |

Class F1 for the frozen NLI is SUPPORT 0.403, CONTRADICT 0.471, and NEI 0.556. The majority baseline's class F1 is SUPPORT 0.585, CONTRADICT 0.000, and NEI 0.000.

Micro precision, recall, and F1 are 0.319, 0.161, and 0.214 for the frozen NLI, and 0.017, 0.014, and 0.015 for the baseline.

Document-level retrieval p95 was 26 ms. The NLI pass scored 2,853 pairs in 110.4 s. 0 pairs were truncated.
<!-- tables:end verdict-dev -->

It always predicts SUPPORT, so that class F1 is higher and the other two are
zero. Macro-F1 is the unweighted mean of the three.

Frozen NLI minus the majority baseline
([verdict-baseline-vs-nli.json](benchmarks/scifact/verdict-baseline-vs-nli.json)):

<!-- tables:begin verdict-paired -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| Accuracy | +0.080 | [-0.013, +0.170] | 0.1125 |
| Macro-F1 | +0.282 | [+0.216, +0.347] | none |
| Sentence precision | +0.393 | [+0.338, +0.448] | 0.0001 |
| Sentence recall | +0.372 | [+0.319, +0.425] | 0.0001 |
| Sentence F1 | +0.373 | [+0.320, +0.425] | 0.0001 |

Above zero: Macro-F1, Sentence precision, Sentence recall, Sentence F1. Intervals that include zero: Accuracy.
<!-- tables:end verdict-paired -->

Macro-F1 was the train selection metric. It is recomputed on each resampled
claim list, so that row has no sign-flip p-value. Sentence precision, recall,
and F1 are means of the per-claim evidence scores against the gold rationale
sentences. Micro scores pool sentence counts. Both-empty claims add nothing.
One PyTorch thread. This comparison file has no `passed` field. It is a
measurement, not a quality gate.

## Judge sample

[benchmarks/support/judge-sample.json](benchmarks/support/judge-sample.json)
judges two questions copied from the support fixture (`api-keys-1`,
`rate-limits-1`). Answers come from the extractive generator. The judge is
`gpt-5-nano`, and the response records `gpt-5-nano-2025-08-07`. Rates in
[configs/support-judge.yaml](configs/support-judge.yaml) are the published
text prices of $0.05 input, $0.005 cached input, and $0.40 output per million
tokens.

<!-- tables:begin judge-sample -->
The report's own totals are 1,236 input tokens, 2,692 output tokens, estimated cost $0.001139, mean correctness 0.500, and mean faithfulness 0.625.
<!-- tables:end judge-sample -->

Scores are divided by the rubric maximum. One answer was the Markdown heading
rather than the expiration sentence, and the judge scored that correctness at
the bottom of the rubric.
There are no human labels and no agreement statistic.

## Capabilities

- **Retrieval:** BM25, exact dense cosine search, min-max weighted hybrid, and
  reciprocal rank fusion; deterministic chunks, IDs, and ranking ties.
- **Evaluation:** document Recall@K, nDCG@K, Precision@K, MAP, and full-ranking
  MRR. Optional `answer_exact_match`, `answer_token_f1`, and
  `context_token_precision` are lexical overlap with a reference or with
  retrieved context. This overlap does not establish factual support.
- **Optional LLM judge:** a separate rubric scores correctness and faithfulness
  from 0 to 4. That rubric is not a SciFact evidence label. The lexical overlap
  metrics are not that score.
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
ragstat evaluate --config configs/dense.yaml --output results/dense.json
ragstat evaluate --config configs/hybrid.yaml --output results/hybrid.json

# Exercise answer evaluation locally with deterministic sentence extraction.
ragstat evaluate --config configs/answers.yaml --db results/runs.sqlite
ragstat history --db results/runs.sqlite

# Local read-only API; interactive API documentation at /docs.
python -m pip install -e '.[api]'
ragstat serve --db results/runs.sqlite
```

For opt-in paid generation and judging, see [providers and scoring](docs/providers.md).
On macOS, run dense benchmarks as fresh CLI processes; see the
[native-library notes](docs/getting-started.md#macos-native-library-behavior).

## Docker

```bash
docker build -t ragstat .
docker run --rm ragstat
# Persist reports and history in a named volume.
docker volume create ragstat-data
docker run --rm -v ragstat-data:/data ragstat evaluate \
  --config configs/answers.yaml --output /data/answers.json --db /data/runs.sqlite
docker run --rm -p 127.0.0.1:8000:8000 -v ragstat-data:/data ragstat \
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
uv run pytest -q --cov=ragstat --cov-report=term-missing --cov-fail-under=85
uv run pytest -q --run-integration
```

Offline tests cover retrieval mathematics, report validation, regression failures,
provider HTTP contracts, judge parsing, storage, and API behavior. Two additional
integration tests use real MiniLM embeddings through the dense and hybrid CLIs.

<!-- tables:begin coverage -->
[benchmarks/pytest-junit.xml](benchmarks/pytest-junit.xml) records 135 tests, 0 failures, and 2 skipped. [benchmarks/test-coverage.json](benchmarks/test-coverage.json) records 1,728 statements, 1,540 covered, 188 missing, and `percent_covered` 89.120.
<!-- tables:end coverage -->

CI fails the job under 85% and also runs mypy. It enforces
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

ragstat is a small, typed retrieval-evaluation library. Neighboring tools
optimize different contracts:

- **Ragas** scores generation and retrieval with LLM-judged metrics on each
  sample. ragstat's primary metrics are deterministic functions of a labeled
  ranking. The optional judge is a separate stage with a versioned rubric,
  strict JSON validation, and caller-supplied token prices.
- **DeepEval** packages LLM metrics as pytest-style checks and can report to a
  hosted product. ragstat keeps the gate local: fingerprints, thresholds, and
  a failing process exit, with a read-only API that cannot start a paid run.
- **promptfoo** is a prompt and assertion matrix, including red-team cases.
  ragstat does not vary prompts. It varies retrievers, chunking, and fusion,
  and it refuses to compare reports whose corpus or question fingerprints differ.
- **ARES** trains or prompts a judge to predict human preference and puts
  intervals around that prediction. ragstat's intervals resample a fixed labeled
  query set. They are not a substitute for human agreement, and this repository
  does not synthesize relevance labels.

## Design limits

Indexes are rebuilt in memory and searched over the full corpus. That keeps
ranking and fusion inspectable. It is the wrong shape for a multi-million
document production index.

The committed BM25 baseline on the 12-document fixture has Recall@3 and
Recall@5 of 1.000, and its document count equals its chunk count, so that
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
