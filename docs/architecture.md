# Architecture decisions

These are the implemented contracts for version 1.0.0. Tests exercise each
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
                            Recall@K + reciprocal rank
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

Use `rank_bm25.BM25Okapi`. Tokenize both corpus and query with Unicode word
matches (`\w+`) after `casefold()`. This tokenizer is separate from the chunk
window definition. Initially there is no stemming or stopword removal.
Reject corpora with no searchable tokens; allow empty token lists for individual
chunks if the corpus contains searchable text elsewhere.

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

Normalizing the whole corpus makes results independent of the requested K and
avoids losing candidates before fusion. This is intentionally a small-corpus
design. A bounded candidate pool can be introduced later with explicit recall
and normalization tradeoffs.

## 9. Retrieve chunks, evaluate documents

The benchmark labels documents. Rank all chunks, then retain the first hit for
each document ID. This is equivalent to ranking each document by its best chunk
score, with ordering inherited from the sorted chunk hits.

Apply K after removing duplicate document IDs. A document with many matching
chunks must not consume several document-ranking positions. The first version
requests the full chunk ranking, so it can calculate document Recall@K and
**full-ranking MRR** without accidentally truncating either metric.

Average each metric equally across questions. Do not micro-average relevance
counts across the dataset. Require at least one relevant document per question
and at least one question per benchmark. A metric called MRR@K would be
truncated; the current metric uses the complete document ranking.

Full-corpus scoring and sorting costs more than bounded retrieval. That is an
accepted first-version limit. The default synthetic fixture contains 12 documents and 27 questions. It checks
engineering behavior, not accuracy on unseen domains. The original three-document
fixture remains available through `configs/smoke.yaml`.

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
See [regression semantics](regression.md).

## 11. Optional generation and judging

`generation/providers.py` defines a small generator protocol, deterministic local
extraction, and a direct HTTP adapter. The runner selects one best chunk per unique
document and bounds total context characters. Reference answers are used only by
metrics and the separate judge. Each stage records its own timing and responses.

Lexical answer metrics live in `evaluation/answers.py`; versioned judge instructions
and strict score validation live in `evaluation/judge.py`. Keeping these separate
makes their different assumptions visible. The provider returns actual token usage
and model identity; prices are configuration data. See [providers](providers.md).

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

The Docker image runs as a non-root user. CI checks supported commands, retrieval
regression gates, model integration, and the container entry point. Indexes remain
in-memory and are rebuilt per run; persistent vector databases and approximate
search would require separate scale-driven tradeoffs.
