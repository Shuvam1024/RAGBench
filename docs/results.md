# Results

This page records how the frozen SciFact and NFCorpus numbers were chosen and what they say.
BM25 (Best Matching 25) is the lexical ranker. nDCG (normalized discounted cumulative gain),
MRR (mean reciprocal rank), and MAP (mean average precision) are ranking metrics.
A CI in a results table is a confidence interval. The CI badge on the repository is
continuous integration. NLI (natural language inference) labels a claim against a sentence.
BEIR (Benchmarking IR) is the corpus collection that supplies SciFact and NFCorpus.

The 300-query SciFact test split and the SciFact dev claims are held out. Settings were
chosen on the train split only. Nothing on this page was selected on test or dev.

## Contents

- [Definitions](#definitions)
- [Train selection](#train-selection)
- [Cross-encoder second stage](#cross-encoder-second-stage)
- [Held-out SciFact test](#held-out-scifact-test)
- [Post-hoc parameter ablation](#post-hoc-parameter-ablation)
- [NFCorpus confirmation](#nfcorpus-confirmation)
- [Exploratory test-split runs](#exploratory-test-split-runs)
- [SciFact claim verdicts](#scifact-claim-verdicts)
- [Held-out SciFact dev](#held-out-scifact-dev)
- [Oracle retrieval and error breakdown](#oracle-retrieval-and-error-breakdown)
- [Judge sample](#judge-sample)

## SciFact

The 300-query test split was used while the original chunk size, tokenizer, and
hybrid weight were still being compared. Those runs stay in this repository as
exploratory results. Revised settings were selected on the 809-query train
split ([ir-datasets `beir/scifact/train`](https://ir-datasets.com/beir.html#beir/scifact/train))
and committed before this test split was scored again. NFCorpus was not used
to choose any setting. Metric figures below are rounded to 3 decimals.
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
| Original chunked | Exploratory test run, not a CI gate | whitespace `\w+` | 120-word windows, overlap 20 | k1 1.5, b 0.75 | `configs/scifact-bm25.yaml` |
| Selected chunked | Chosen on train, scored once on test, CI gate | stem | 480-word windows, overlap 20 | k1 1.5, b 0.75 | `configs/scifact-bm25-selected.yaml` |
| Document-level | Pre-specified baseline and CI gate | stem | one string per document | k1 0.9, b 0.4 | `configs/scifact-bm25-document.yaml` |
| Document-level, selected parameters | Post-hoc ablation after the test report, not a selection | stem | one string per document | k1 1.5, b 0.75 | `configs/scifact-bm25-document-k1.5-b0.75.yaml` |
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
three, so it is the frozen depth. The test split is scored once after that choice is
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
Paired deltas use seed 0, 10,000 bootstrap resamples, 10,000 sign-flips, and a 95% percentile interval. Committed retrieval comparison files do not store that recipe. A newly written comparison does, when its thresholds set statistics. The recipe matches `configs/paired-uncertainty.yaml` and the verdict comparison.
<!-- tables:end paired-settings -->
Positive delta means the candidate is higher. That file sets `max_drop` to 1.0
so the comparison records the interval. `passed` in those files is a measurement
flag. Continuous integration gates the frozen document-level BM25 config and
the frozen selected BM25 config with a zero point-drop in
`configs/scifact-thresholds.yaml`. The original chunk-120 report is exploratory.

<!-- tables:begin chunk-window -->
Only 46 of 5,183 SciFact documents and 26 of 3,633 NFCorpus documents are longer than 480 words, so the selected chunked index is nearly one chunk per document. A post-hoc ablation, scored after the test numbers were reported and not used to select a setting, uses that document-level index with parameters 1.50/0.75. Its test nDCG@10 is 0.6910. Document-level BM25 at 0.90/0.40 is 0.674, and the parameter change is +0.017. The selected chunked index scores 0.6907. Those two differ on 5 of 300 queries, so chunking adds about nothing on SciFact once the parameters match. On train the frozen pair tie at 0.6972 vs 0.6957.
<!-- tables:end chunk-window -->

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

### Post-hoc parameter ablation

This comparison was scored after the test numbers above were already reported.
It is not a train selection, and it does not change the frozen document-level
baseline or the selected chunked index. The new index is document-level BM25
with the selected parameters (1.50 and 0.75) and the same stem tokenizer
(`configs/scifact-bm25-document-k1.5-b0.75.yaml`).

Document-level BM25 at those parameters minus the frozen document-level baseline
([document-vs-bm25-k15.json](benchmarks/scifact/document-vs-bm25-k15.json)):

<!-- tables:begin scifact-doc-vs-k15 -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.017 | [+0.005, +0.028] | 0.0045 |
| Recall@10 | +0.037 | [+0.016, +0.061] | 0.0006 |
| Recall@100 | -0.003 | [-0.010, +0.000] | 1.0000 |
| MAP | +0.008 | [-0.003, +0.020] | 0.1765 |
| MRR | +0.010 | [-0.003, +0.023] | 0.1522 |

Above zero: nDCG@10, Recall@10. Intervals that include zero: Recall@100, MAP, MRR.

Per-query nDCG@10 differs on 51 of 300 queries.
<!-- tables:end scifact-doc-vs-k15 -->

Selected chunked BM25 minus that ablation
([bm25-k15-vs-selected.json](benchmarks/scifact/bm25-k15-vs-selected.json)).
Positive delta would mean chunking scored higher:

<!-- tables:begin scifact-k15-vs-selected -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.000 | [-0.001, +0.000] | 0.0621 |
| Recall@10 | +0.000 | [+0.000, +0.000] | 1.0000 |
| Recall@100 | +0.000 | [+0.000, +0.000] | 1.0000 |
| MAP | +0.000 | [-0.001, +0.000] | 0.0222 |
| MRR | +0.000 | [-0.001, +0.000] | 0.0166 |

Below zero: nDCG@10, MAP, MRR. Intervals that include zero: Recall@10, Recall@100. nDCG@10 is below zero, but its upper bound rounds to +0.000. MAP is below zero, but its upper bound rounds to +0.000. MRR is below zero, but its upper bound rounds to +0.000.

Per-query nDCG@10 differs on 5 of 300 queries.
<!-- tables:end scifact-k15-vs-selected -->

The nDCG@10 interval is just below zero, and its sign-flip p-value does not
fall under 0.05. MAP and MRR are slightly lower. At the displayed precision
the chunking deltas are 0.000, so the gap versus the frozen document-level
baseline is the parameter change, not the 480-word windows.

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
| Recall@10 | +0.020 | [+0.000, +0.043] | 0.0600 |
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
| MAP | +0.000 | [-0.003, +0.002] | 0.8801 |
| MRR | +0.011 | [-0.003, +0.026] | 0.1207 |

Above zero: nDCG@10. Intervals that include zero: Recall@10, Recall@100, MAP, MRR. nDCG@10 is above zero, but its lower bound rounds to +0.000.
<!-- tables:end nfcorpus-doc-vs-selected -->

That does not confirm a general advantage for the 480-word stem index over
document-level BM25.

The same post-hoc ablation, document-level BM25 with parameters 1.50 and 0.75,
was scored on this corpus after the frozen numbers were reported. It was not
used to choose a setting
(`configs/nfcorpus-bm25-document-k1.5-b0.75.yaml`).

That ablation minus the frozen document-level baseline
([document-vs-bm25-k15.json](benchmarks/nfcorpus/document-vs-bm25-k15.json)):

<!-- tables:begin nfcorpus-doc-vs-k15 -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.004 | [-0.001, +0.008] | 0.0975 |
| Recall@10 | +0.000 | [-0.004, +0.004] | 0.8648 |
| Recall@100 | +0.001 | [-0.001, +0.004] | 0.4209 |
| MAP | +0.000 | [-0.004, +0.002] | 0.8198 |
| MRR | +0.010 | [-0.005, +0.024] | 0.1819 |

Intervals that include zero: nDCG@10, Recall@10, Recall@100, MAP, MRR.

Per-query nDCG@10 differs on 114 of 323 queries.
<!-- tables:end nfcorpus-doc-vs-k15 -->

Selected chunked BM25 minus that ablation
([bm25-k15-vs-selected.json](benchmarks/nfcorpus/bm25-k15-vs-selected.json)):

<!-- tables:begin nfcorpus-k15-vs-selected -->
| Metric | Mean delta | 95% CI | Sign-flip p |
| --- | --- | --- | --- |
| nDCG@10 | +0.001 | [+0.000, +0.003] | 0.0789 |
| Recall@10 | +0.003 | [+0.000, +0.007] | 0.0273 |
| Recall@100 | -0.002 | [-0.010, +0.005] | 0.7118 |
| MAP | +0.000 | [+0.000, +0.001] | 0.6297 |
| MRR | +0.002 | [+0.000, +0.004] | 0.0462 |

Above zero: Recall@10, MRR. Intervals that include zero: nDCG@10, Recall@100, MAP. Recall@10 is above zero, but its lower bound rounds to +0.000. MRR is above zero, but its lower bound rounds to +0.000.

Per-query nDCG@10 differs on 12 of 323 queries.
<!-- tables:end nfcorpus-k15-vs-selected -->

On this corpus the parameter change does not clear zero, and neither does the
nDCG difference between the chunked index and the matched document-level
parameters. Recall@10 is slightly higher for the chunked index.

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
300 labeled claims. Gold counts are SUPPORT 124, CONTRADICT 64, and NEI 112. Micro sentence F1 is 0.214 for the frozen NLI and 0.015 for the majority baseline. Evidence-only sentence F1, the mean over claims with nonempty gold evidence, is 0.193 vs 0.018. The per-claim mean in the table below is 0.384 vs 0.012. That mean scores a both-empty evidence set as a perfect match, which is why it sits above the micro and evidence-only figures.

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


## Oracle retrieval and error breakdown

<!-- tables:begin verdict-oracle -->
| Slice | Count |
| --- | --- |
| SUPPORT predicted NEI | 67 |
| of those, gold document inside top 1 | 47 |
| of those, gold document missed | 20 |
| Incorrect claims with a gold document in the top 1 | 78 |
| Incorrect claims whose gold document was missed | 41 |
| Incorrect claims with no gold document | 33 |

The oracle gives the frozen sentence policy every gold evidence document and does not change sentence_k or min_confidence. It was not selected on dev. Its macro-F1 is 0.603 and its micro sentence F1 is 0.302. Evidence-only sentence F1 is 0.247. The frozen retrieved NLI system, on the same claims, has macro-F1 0.477 and micro sentence F1 0.214.
<!-- tables:end verdict-oracle -->

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
