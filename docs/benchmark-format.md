# Benchmark format and metric examples

This is the implemented version 1 contract. Pydantic validates the input before
the runner builds an index. Models live in `ragbench/evaluation/models.py`.

## JSON structure

```json
{
  "schema_version": 1,
  "questions": [
    {
      "id": "dense-search",
      "question": "Which tools create embeddings and search vectors for dense retrieval?",
      "expected_answer": "A sentence-transformers model creates embeddings and FAISS searches the vectors.",
      "relevant_document_ids": ["retrieval.txt"]
    }
  ]
}
```

- `id`: a nonblank question identifier, unique within the benchmark.
- `question`: the nonblank search query passed to the retriever.
- `expected_answer`: reference text for answer scoring and judging. It may be
  empty for a retrieval-only benchmark. Generation refuses a blank reference.
- `relevant_document_ids`: a nonempty list of distinct document IDs in the corpus.
- `relevance_grades`: optional map from those document IDs to positive grades.
  Omitted grades are binary (every relevant document has grade 1) and are not
  stored, so adding the field as null does not change the benchmark fingerprint.

Document IDs are paths relative to `dataset.documents_path`, using `/` as the
separator and preserving case. For `datasets/documents/guides/setup.md`, the ID
is `guides/setup.md`. It is not an absolute path or a generated chunk ID.

Before building an index, the runner rejects duplicate question IDs,
unknown document references, duplicate relevance labels, blank fields, unknown
fields, unsupported schema versions, and empty question lists. Validation
should name the question or field causing the problem.

The expected answer and relevance labels must never be provided to a retriever
as input. Only the question is used for searching.

## Recall@K

For question `q`, let `R` be its relevant document IDs and `D[:K]` the first K
unique retrieved document IDs:

```text
Recall@K(q) = |R intersect D[:K]| / |R|
```

Example: relevant documents are `{A, C}` and the document ranking is `[B, C, A]`.

- Recall@1 = 0/2 = 0.
- Recall@2 = 1/2 = 0.5.
- Recall@3 = 2/2 = 1.

The reported Recall@K is the arithmetic mean of per-question recall values.
If K exceeds the ranking length, use all available results; keep the denominator
equal to the total number of relevant documents. An empty ranking scores zero.
An empty relevance set is invalid, because the denominator would be zero.

## Precision@K and average precision

Precision@K is the fraction of the top K unique document slots that are relevant.
The denominator is K, even when the question has fewer than K relevant documents.
On a corpus with about one relevant document per question, Precision@10 therefore
cannot exceed about 0.1.

Average precision walks the full unique-document ranking. Each time a relevant
document appears at rank `r` as the `h`-th hit, it contributes `h / r`. Divide by
the number of relevant documents. MAP is the unweighted mean of those values.
The JSON field is `mean_average_precision`; the threshold name is `map`.

## nDCG@K

Discounted cumulative gain at K is the sum of `(2^grade - 1) / log2(rank + 1)`
over the top K unique documents. Unlisted documents have grade 0. nDCG divides
that sum by the ideal DCG of the highest grades, also truncated at K. Binary
relevance is grade 1. Graded qrels are accepted when `relevance_grades` is set.

## Reciprocal rank and MRR

```text
RR(q) = 1 / rank of the first relevant document, or 0 if none is retrieved
MRR = sum(RR(q) for each question) / number of questions
```

For `[B, C, A]` with relevant documents `{A, C}`, the first relevant result is
at rank 2, so RR = 1/2. A second question whose first relevant result is at rank
1 has RR = 1. Together their MRR is `(0.5 + 1) / 2 = 0.75`.

Ranks start at one. MRR uses the full returned document ranking in this first
version. It is separate from the configured Recall@K cutoffs.

## Why collapse chunk hits first?

Suppose chunk hits belong to documents `[B, B, C, A]`. Collapse them to
`[B, C, A]` before calculating document-level metrics. If C is relevant, its
document rank is 2. Returning several chunks from B does not give B several
document slots.

## Result JSON

The CLI supports `--output results/baseline.json`, and YAML supports
`output.json_path`. The result envelope includes:

- Schema version and effective configuration.
- Corpus/benchmark fingerprints and document/chunk/question counts.
- Retrieval implementation and dependency/model versions.
- For each question: relevant IDs, retrieved document IDs and best-chunk scores,
  Recall@K values, and reciprocal rank.
- Aggregate Recall@K, nDCG@K, Precision@K, MAP, and full-ranking MRR.
- `multi_chunk_documents`: how many documents produced more than one chunk.
- `stored_hits`: how many retrieved documents are written per question. Metrics
  still use the full ranking.

The core top-level fields include `schema_version`, `config`, `corpus_sha256`,
`benchmark_sha256`, `document_count`, `chunk_count`, `question_count`,
`skipped_empty_files`, `retriever`, `versions`, `recall_at_k`, `mrr`, and `questions`.
Each question includes `id`, `question`, `relevant_document_ids`,
`retrieved_documents`, `recall_at_k`, and `reciprocal_rank`. A retrieved document
contains `document_id`, `chunk_id`, and `score` for its best-ranked chunk.

Recall cutoff keys become strings in JSON. Scores are comparable within a
retriever's ranking; a BM25 raw score and a dense cosine score have different
scales. Hybrid reports its normalized weighted scores. The CLI exports actual
measurements; disabled answer stages and unknown provider prices remain null.

Fingerprints use SHA-256 over compact, sorted-key JSON. The corpus fingerprint
covers sorted pairs of document IDs and normalized-text hashes. The benchmark
fingerprint covers the validated benchmark, including reference answers. Changing
an expected answer therefore changes provenance even though it does not affect
retrieval scores. Reports include local absolute data/output paths; review those
paths before sharing a generated report.


Report schema 2 additionally records `run_id`, `created_at`, `timings`, and
`environment`. Optional fields include `answer_metrics`, `judge_metrics`,
`input_tokens`, `output_tokens`, and `estimated_cost_usd`. Each question can include
an answer, context chunk IDs, judge scores and explanation, and stage timings.
See the Pydantic models for the complete machine-readable field definitions and
[providers](providers.md) for metric semantics.
