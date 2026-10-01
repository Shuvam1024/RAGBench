# Measuring retrieval quality

A benchmark question identifies the documents that should be retrieved.
These relevance labels let us evaluate retrieval without generating an answer.

Recall@K is the number of relevant documents found in the first K retrieved
documents divided by the total number of relevant documents for the question.
If two documents are relevant and one appears in the first three results,
Recall@3 is one half.

Reciprocal rank is one divided by the rank of the first relevant document.
If the first relevant result is third, reciprocal rank is one third. If no
relevant result appears, reciprocal rank is zero. Mean Reciprocal Rank, or MRR,
is the average reciprocal rank across benchmark questions.

Multiple chunks from one document are collapsed before document-level metrics
are calculated. Expected answers are reserved for later answer evaluation.
