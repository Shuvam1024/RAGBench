# Generation, judging, and cost accounting

Retrieval-only runs are the default. `configs/answers.yaml` adds a deterministic
extractive generator that selects a context sentence by query-term overlap. It
exercises the answer pipeline locally, without claiming LLM reasoning ability.

## Optional OpenAI adapter

```bash
python -m pip install -e '.[llm]'
cp configs/openai.example.yaml configs/openai.local.yaml
# Edit the model names in the copied configuration.
# Set OPENAI_API_KEY securely in your shell environment.
ragbench evaluate --config configs/openai.local.yaml --output results/openai.json
```

Select available model IDs compatible with the Responses API and structured
JSON output for judging. The example deliberately has placeholders. Generation
and judge models can differ. Configure the judge only when you want its additional
API calls and cost. Remove `judge` to run generation alone. Do not put API keys in
YAML or Git. Evaluation sends benchmark questions and selected context to the
provider; judging additionally sends reference answers and generated answers.

The direct HTTP adapter uses the [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create),
requests `store: false`, limits output tokens, and applies a bounded request timeout.
It has no automatic retry loop. Refusals, incomplete responses, missing usage, HTTP
errors, and malformed judge JSON fail the run. A failed run does not publish a
complete report, though earlier provider requests can still incur charges.

## Context and answer metrics

The generator receives the best chunk from each of the first `context_k` unique
documents, bounded by `max_context_chars` across all chunks. The final chunk may be
truncated. Reference answers never enter the generator prompt. Reports retain the
selected chunk IDs; corpus fingerprints and configuration identify source inputs.

- `answer_exact_match`: case-folded Unicode word sequences match exactly and the
  answer is nonempty. Punctuation differences are ignored.
- `answer_token_f1`: multiset token precision/recall harmonic mean against the
  reference; repeated tokens count. Empty answers score zero.
- `context_token_precision`: fraction of answer tokens present in retrieved
  context. This is a lexical overlap proxy, not proof of faithfulness.

The extractive baseline naturally has high context overlap because it copies a
sentence. Its low reference match is expected; do not present that overlap as
hallucination detection accuracy.

## Versioned judge rubric

`ragbench-judge-v1` asks for integer scores from 0 to 4 and a short explanation:

- **Correctness:** 0 wrong/unanswered, 1 mostly wrong, 2 partially correct with
  major omissions, 3 mostly correct with minor omissions, 4 fully correct.
- **Faithfulness:** 0 unsupported/contradictory, 1 mostly unsupported, 2 mixed,
  3 supported except minor claims, 4 all material claims supported.

Correctness uses the reference. Faithfulness uses only retrieved context.
Abstention is faithful only when that context lacks the required evidence.
Aggregate `judge_correctness` and `judge_faithfulness` divide each score by four
and average across questions. Scores include the rubric version, actual returned
model identity, explanation, usage, and estimated cost.

A strict JSON schema and Pydantic validation reject missing/extra fields,
booleans, non-integer scores, and values outside the rubric. Prompt instructions
treat supplied fields as untrusted data; this is not a guarantee against prompt
injection. The tests verify protocol behavior and validation using mock HTTP
responses. No live paid judge run or human-label calibration is claimed. Model
bias, variability, and reference quality remain limitations. Pin model versions
and calibrate against human judgments before using judge gates for release policy.

## Tokens, cost, and timing

Set `pricing` separately for each provider configuration, in USD per million
tokens: `input_per_million`, `cached_input_per_million`, `output_per_million`.
No prices are inferred from model names. Cost is:

```text
((input - cached_input) × input_rate
 + cached_input × cached_rate
 + output × output_rate) / 1,000,000
```

Usage and estimated cost aggregate generation and judging. Cost is null when any
provider response lacks pricing; null means unknown. Local extraction costs zero
API dollars and has no API token count. Local embedding compute and hardware are
excluded. Estimates depend on your supplied rates and provider usage, and are not
billing statements.

Index timing includes model initialization and any first-run download. Retrieval
latency includes query encoding and full-corpus retrieval. Generation and judge
latencies cover their respective calls. Percentiles use linear interpolation over
one pass through benchmark questions; there is no warm-up exclusion or load test.
