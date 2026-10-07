# Architecture decisions

These are the implemented contracts for version 1.1.0. Tests exercise each
boundary independently and through the CLI.

## 1. A small, synchronous pipeline

```text
YAML configuration + benchmark questions
                 |
                 v
UTF-8 files -> Documents -> Chunks -> Retriever index
                                         |
                                      Question
                                         |
                                         v
                                 Ranked chunk hits
                                         |
                              Collapse document IDs
                                         |
                                         v
                  Recall@K + nDCG@K + precision@K + MAP + MRR
                                         |
                              Average across questions
                                         |
                               Terminal / JSON report
```

`config.py` owns configuration parsing and validation. `loader.py` owns files
and documents. `chunker.py` owns text windows. Retriever modules own scoring.
`evaluation/retrieval.py` contains pure metric functions with no I/O.
`runner.py` connects these pieces; `evaluation/models.py` owns benchmark/result models. `cli.py`
handles user-facing options, terminal output, and writing JSON results.

Use Pydantic for configuration, benchmark, result, document, and chunk models.
Keep models in their owning modules until a separate shared module is justified.
The dependency direction is CLI → runner → ingestion/retrieval/metrics.
Lower-level modules must not import the CLI or runner.

## 2. Configuration is an experiment contract

Use Pydantic v2 models with unknown fields forbidden. Use strict numeric fields
so booleans and numeric strings cannot silently become chunk sizes or weights.
YAML strings representing paths are explicitly accepted and converted to paths.
Use `yaml.safe_load` and reject non-mapping configuration roots.

`retrieval.type` selects `bm25`, `dense`, or `hybrid` through a discriminated
union. Each mode accepts only its own settings. Hybrid contains BM25 and dense
settings plus `dense_weight`; dense settings include a model name, optional
revision, batch size, and CPU device by default.

All paths inside YAML resolve against the YAML file's parent. The CLI's
`--config` and `--output` arguments resolve against the shell's current
directory. An explicit `--output` overrides `output.json_path`.
Null output paths mean terminal output only. Create output parent directories
when saving; reject a destination that would overwrite a source document,
benchmark, or configuration. Saving to an existing result file replaces it.

The [Pydantic model documentation](https://pydantic.dev/docs/validation/latest/concepts/models/)
describes typed models and rejecting extra fields.

## 3. Documents have portable, readable identities

- Recursively discover `.txt` and `.md` files, with suffix matching independent
  of case. Sort by the relative POSIX path. Skip symlinks.
- Read UTF-8, accepting an optional byte-order mark. Normalize CRLF and CR to LF.
- Keep Markdown as text; do not render it or remove markup.
- Set the document ID to its exact relative POSIX path, such as
  `guides/retrieval.md`. The documents-root directory is excluded.
- Store a SHA-256 fingerprint of the normalized text separately from the ID.
- Skip whitespace-only files and report the skipped count. Fail if there are
  no usable documents. Invalid UTF-8 and unreadable files report their paths.

Moving an unchanged corpus to another machine preserves its document IDs.
Renaming a file changes its ID. Editing its contents preserves the document ID
but changes the content fingerprint, which lets us detect changed inputs.
Files with identical content and different paths remain distinct documents.

A directory that contains `documents.jsonl`, or a path that is itself a
`.jsonl` file, is loaded as JSONL instead of a text walk. Each line is one
object with `id` and `text`. IDs are sorted lexicographically. This is how the
SciFact materializer hands the corpus to the same loader the file walk uses.

## 4. Chunking counts whitespace-delimited words

For the first version, a word is a non-whitespace span (`\S+`). This gives a
clear, model-independent definition of chunk size. These are not model tokens.

For size `S` and overlap `O`, require `S > 0` and `0 <= O < S`. Start at word
zero and advance by `S - O`. Emit the final partial window and stop as soon as
a window reaches the end. Do not emit an extra overlap-only tail.

Preserve the original text between the first and last word spans in a window,
including internal newlines. A chunk records its ID, parent document ID, text,
and start/end word offsets; the end offset is exclusive.

Example: `a b c d e f g h`, size 5, overlap 2:

```text
window 0: [0, 5) -> a b c d e
window 1: [3, 8) ->       d e f g h
```

Chunk IDs use `chunk-v1-` plus the full SHA-256 hex digest of the UTF-8 JSON
encoding of this ordered array:

```text
[document_id, document_content_sha256, chunk_size, overlap, start_word, end_word]
```

Use compact JSON separators and `ensure_ascii=False` for a fixed encoding.
Identical inputs and settings produce identical IDs. Content edits or chunking
changes produce new chunk IDs. Do not use Python's process-dependent `hash()`.

Long words and model subword tokenization mean a word window can still exceed
an embedding model's limit. At the dense step, detect and report truncated
chunks rather than silently treating all text as encoded. The initial baseline
uses 120 words and 20 words of overlap; this is a teaching default, not a tuned
setting.

## 5. One retriever interface

The abstract interface in `retrieval/base.py` exposes:

```python
index(chunks: Sequence[Chunk]) -> None
retrieve(query: str, k: int) -> list[SearchResult]
```

`SearchResult` contains the chunk and a finite floating-point score. Higher is
better. Sort by descending score and then ascending chunk ID for ties. Indexing
replaces prior state. Retrieving before indexing or with `k <= 0` is an error;
blank queries return no hits. Oversized K returns the available chunks.
Reject an empty index and duplicate chunk IDs.

Do not couple any retriever to benchmark labels or expected answers. The same
interface must be usable independently of evaluation.

## 6. BM25 is the first baseline

Use `rank_bm25.BM25Okapi`. The default tokenizer, `whitespace`, splits both the
corpus and the query with Unicode word matches (`\w+`) after `casefold()`.
`tokenizer: stem` uses that same split, drops a vendored English stopword list,
and applies the Snowball English stemmer from the pure-Python `snowballstemmer`
package. The default stays `whitespace` so older configs keep their scores.
This tokenizer is separate from the chunk window definition. Reject corpora
with no searchable tokens; allow empty token lists for individual chunks if
the corpus contains searchable text elsewhere. A query that is only stopwords
returns no hits under the stem tokenizer.

`chunking.unit: words` is the overlapping word-window index. `chunking.unit:
document` indexes each prepared document as one string. A document-level run
is a different experiment from the chunked run. Neither setup is Lucene or the
Pyserini BEIR flat index, even when `k1` and `b` are set to Pyserini's `--bm25`
defaults of 0.9 and 0.4.

Keep library score values, including zero and negative values. A nonblank query
whose terms do not appear in the corpus produces tied scores and therefore a
deterministic ranking; a query with no lexical tokens returns no hits. Tests
must not assume that every positive word match has a positive BM25 score.

The [rank-bm25 README](https://github.com/dorianbrown/rank_bm25/blob/master/README.md)
requires callers to tokenize the corpus and queries consistently.

## 7. Dense search is exact and testable

The starting model is `sentence-transformers/all-MiniLM-L6-v2`, with its resolved
revision recorded for actual benchmark runs. Example YAML files pin the tested
revision. Model selection stays configurable.
Use CPU initially, float32 embeddings, and L2 normalization for both chunks and
queries. FAISS `IndexFlatIP` then ranks by cosine similarity. Reject invalid
dimensions, nonfinite vectors, and zero-norm vectors with useful errors.

Use exact search for the small corpus so approximate-index tuning does not
complicate the lesson. Deterministically break ties before taking the final K;
FAISS's internal tie order is not our public contract.

Import dense libraries only when dense/hybrid retrieval is selected. Load the
model once per retriever. `retrieval/embeddings.py` owns the embedding-provider
protocol and Sentence Transformers adapter. Offline unit tests inject known
vectors while still exercising the real FAISS index.

The adapter sets PyTorch inference to one CPU thread and records this setting.
It counts inputs exceeding the tokenizer limit, warns before encoding, and
reports `truncated_texts` in retriever metadata. This counts both chunks and
queries encoded by that provider.

On the tested macOS environment, initializing both native OpenMP runtimes in
different orders can still crash a long-lived Python process. Run model-based
benchmarks as fresh CLI processes. Integration tests use the same process boundary;
see [the platform notes](getting-started.md#macos-native-library-behavior).

References: [Sentence Transformers quickstart](https://sbert.net/docs/quickstart.html)
and [FAISS cosine similarity](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

## 8. Hybrid combines comparable score ranges

For each question, get a score for every indexed chunk from each retriever.
Both score vectors must cover exactly the same chunk IDs. Normalize each vector
over the full corpus:

```text
normalized_score = (score - minimum_score) / (maximum_score - minimum_score)
hybrid_score = (1 - dense_weight) * bm25_normalized + dense_weight * dense_normalized
```

If a score vector is constant, its normalized values are all zero: it provides
no ranking information. An empty lexical query contributes a zero BM25 vector
to hybrid scoring. Require `0 <= dense_weight <= 1`; start at `0.5`.
This weight means the dense contribution; zero selects lexical ranking and
one selects dense ranking. Ties use the shared chunk-ID ordering.

`fusion: rrf` is the other hybrid option. Reciprocal rank fusion ignores
`dense_weight` and scores each chunk as the sum of `1 / (rrf_k + rank)` over
the two full rankings. The default `rrf_k` is 60. Chunk-ID tie breaks happen
before ranks are assigned, so tied raw scores still receive distinct ranks.
An empty lexical ranking contributes zeros, the same as min-max fusion.

Normalizing or fusing the whole corpus makes results independent of the
requested K and avoids losing candidates before fusion. This is intentionally
a small-corpus design. A bounded candidate pool can be introduced later with
explicit recall and normalization tradeoffs.

## 9. Retrieve chunks, evaluate documents

The benchmark labels documents. Rank all chunks, then retain the first hit for
each document ID. This is equivalent to ranking each document by its best chunk
score, with ordering inherited from the sorted chunk hits.

Apply K after removing duplicate document IDs. A document with many matching
chunks must not consume several document-ranking positions. Metrics are
computed on the full document ranking. `evaluation.stored_hits` then keeps
only the first K retrieved documents in the JSON report, so a large corpus
does not write every chunk score. MRR and MAP still use the complete ranking.

Recall@K is `|relevant ∩ top K| / |relevant|`. Precision@K uses K as the
denominator, not the number of relevant documents. Average precision walks
the full unique-document ranking and MAP is the unweighted mean of those
values. nDCG@K uses gains `(2^grade - 1) / log2(rank + 1)`. When a question
has no `relevance_grades`, every relevant document has grade 1. SciFact's
published test qrels are all grade 1, so that run's nDCG is binary even
though the metric accepts graded labels.

Average each metric equally across questions. Do not micro-average relevance
counts across the dataset. Require at least one relevant document per question
and at least one question per benchmark. A metric called MRR@K would be
truncated; the current MRR uses the complete document ranking.

Full-corpus scoring and sorting costs more than bounded retrieval. That is an
accepted first-version limit. `retrieval_p95_ms` is this full-ranking time:
the retriever scores every chunk, including when a later stage only keeps the
top documents. The default synthetic fixture contains 12 documents and 27
questions. It checks engineering behavior, not accuracy on unseen domains.
The original three-document fixture remains available through `configs/smoke.yaml`.

Optional `rerank` settings add a cross-encoder second stage. The first stage
still produces the full document ranking. The cross-encoder then reorders the
first `candidate_k` documents and the reported nDCG, MRR, MAP, and Recall are
computed on that reordered list alone. Documents outside the candidate set are
not returned. `candidate_recall_at_100` is separate: it is the first stage's
Recall@100, measured before the cross-encoder runs. When `candidate_k` is 100,
reranked Recall@100 matches that number because reranking only permutes the
same documents.

A multi-chunk document contributes one string: the text of its highest-scoring
first-stage chunk. Other chunks are not concatenated and are not scored on
their own. Pairs longer than `rerank.max_length` tokens, counting special
tokens, are truncated. The model name and revision are recorded on the report.
The score is the model's raw output. For `cross-encoder/ms-marco-MiniLM-L-6-v2`
that activation is the identity, so the value is a ranking logit.

`index_ms` is the first-stage index build. It does not include cross-encoder
weight loading. Embedding-model initialization still sits inside that index
build, because the embedder is constructed while the first-stage index is
built. `rerank_p95_ms` is the cross-encoder alone. `pipeline_p95_ms` adds the
full-ranking first stage and the cross-encoder. Min-max hybrid normalizes over
every chunk, so this pipeline does not have a cheaper early-exit first stage.
The full-ranking number stays in `retrieval_p95_ms`.

## 10. Results and comparison

YAML and benchmark inputs use schema 1; reports use schema 2. Reports record a
run UUID, UTC timestamp, effective settings, normalized input fingerprints,
library/model versions, per-question evidence, timings, and optional answer data.
Ranking tie rules and sorted input paths support reproducibility; floating-point
scores can still vary across hardware and library versions. Timestamps and timings
are expected to differ between repeated runs.

`evaluation/comparison.py` validates compatible inputs and checks absolute quality
drops and relative resource increases. Missing measurements fail rather than pass.
Aggregate retrieval and answer/judge scores must match their per-question values.
Optional `statistics` settings add a seeded paired bootstrap percentile interval
and a two-sided sign-flip permutation test on per-question deltas. `gate_mode`
names the quality decision. `point_drop` uses the point estimate and is the CI
policy (`gate_on_ci: false`). `proven_regression` fails only when the upper
confidence bound of `(candidate - baseline)` is below `-tolerance`.
`non_inferior` fails unless the lower bound is at least `-tolerance`.
`gate_on_ci: true` is the older spelling of `proven_regression`. The comparison
also lists questions that improved, regressed, or stayed unchanged on each gated
quality metric. See [regression semantics](regression.md).

## 11. Optional generation and judging

`generation/providers.py` defines a small generator protocol, deterministic local
extraction, and a direct HTTP adapter. The runner selects one best chunk per unique
document and bounds total context characters. Reference answers are used only by
metrics and the separate judge. Each stage records its own timing and responses.

Lexical answer metrics live in `evaluation/answers.py`. They are token overlap.
`context_token_precision` does not establish factual support, and it is not a
faithfulness score. Versioned judge instructions and strict score validation
live in `evaluation/judge.py`. The judge rubric is a separate 0–4 model score,
not a SciFact SUPPORT or CONTRADICT label. Keeping these separate makes their
different assumptions visible. The provider returns actual token usage and model
identity; prices are configuration data. See [providers](providers.md).

## 12. Persistence and delivery

`storage.py` stores complete report JSON plus summary columns in SQLite. Each save
is transactional, and duplicate run IDs fail. Read connections use SQLite read-only
mode so a missing database is never created as a side effect. Database schema uses
`PRAGMA user_version = 1`; incompatible versions fail clearly.

The CLI writes SQLite before optional JSON. Each output is atomic independently;
there is no transaction spanning SQLite and the filesystem. If JSON writing fails,
the successfully saved database run remains available.

`api.py` exposes only health, run listing, and run detail. The service defaults to
loopback and provides interactive OpenAPI documentation. Evaluation remains a CLI
operation, so HTTP requests cannot trigger provider spending. There is no multi-user
authentication or hosted deployment in this release.

The Docker image runs as a non-root user and installs from `uv.lock`. CI checks
supported commands, the support-fixture gate, a checksum-verified SciFact BM25
gate, model integration, type checking, coverage, and the container entry point.
Indexes remain in-memory and are rebuilt per run; persistent vector databases and
approximate search would require separate scale-driven tradeoffs.

## 13. Public BEIR materialization

`ragstat dataset scifact` downloads the SciFact zip, checks its SHA-256, and
writes `documents.jsonl` plus `benchmark.json` under the cache directory. Title
and body are joined with a newline when both are non-empty. Query identifiers
are sorted numerically when every ID is digits. Relevance grades are written
only when some qrel score is not 1. The command also writes a corpus-length
manifest. FiQA and NFCorpus are not wired up: SciFact is large enough for
chunking to split thousands of documents, and a 57k-passage corpus would
multiply CPU embedding time without changing the method.

## 14. SciFact verdicts use the claim files

`evaluation/factual.py` scores SUPPORT, CONTRADICT, and NEI plus evidence
sentences. `datasets/scifact_claims.py` reads the AllenAI claim files and checks
them against BEIR query IDs. BEIR qrels stay retrieval grades. `evaluation/nli.py`
loads a pinned NLI model, and tests can inject a scorer. `evaluation/verdict.py`
selects the decision rule on the train claims. The frozen rule is `doc_k` 1,
`sentence_k` 2, and `min_confidence` 0.7. The public SciFact test labels
are withheld, so the held-out labeled split is the dev claims. Dev is scored
once after that choice.
