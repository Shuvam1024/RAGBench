"""Provider contracts are tested without credentials or paid network calls."""

import json
from pathlib import Path

import httpx
import pytest

from ragbench.config import JudgeConfig, OpenAIConfig, Pricing, load_config
from ragbench.evaluation.answers import score_answer
from ragbench.evaluation.judge import LLMJudge
from ragbench.evaluation.runner import evaluate
from ragbench.generation.models import Usage
from ragbench.generation.providers import (
    ContextSource,
    ExtractiveGenerator,
    OpenAIGenerator,
    ResponsesClient,
    estimated_cost,
)

SOURCE = ContextSource(
    document_id="policy.md", chunk_id="chunk", text="Backups expire after seven days."
)


def response(text: str = "Seven days.") -> dict[str, object]:
    return {
        "status": "completed",
        "model": "resolved-model-version",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 10,
        },
    }


def test_cost_and_invalid_usage() -> None:
    usage = Usage(input_tokens=100, cached_input_tokens=20, output_tokens=10)
    prices = Pricing(input_per_million=2.0, cached_input_per_million=1.0, output_per_million=8.0)
    assert estimated_cost(usage, prices) == pytest.approx(0.00026)
    assert estimated_cost(usage, None) is None
    for raw in (
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 0},
    ):
        with pytest.raises(ValueError):
            Usage.model_validate(raw)


def test_generation_request_has_no_reference_and_records_usage() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert body["store"] is False
        assert body["max_output_tokens"] == 512
        assert set(json.loads(body["input"])) == {"question", "context"}
        assert "untrusted" in body["instructions"]
        return httpx.Response(200, json=response())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = OpenAIGenerator(
            OpenAIConfig(model="example"), ResponsesClient("test", client)
        ).generate("When do backups expire?", [SOURCE])
    assert result.text == "Seven days."
    assert result.model == "resolved-model-version"
    assert result.usage == Usage(input_tokens=100, cached_input_tokens=20, output_tokens=10)
    assert result.estimated_cost_usd is None


@pytest.mark.parametrize(
    "raw",
    [
        {"status": "incomplete"},
        {"status": "completed", "output": []},
        {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
        },
    ],
)
def test_incomplete_refused_or_missing_usage_fails(raw: dict[str, object]) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))
    ) as client:
        with pytest.raises(ValueError):
            ResponsesClient("test", client).complete(
                OpenAIConfig(model="example"), "instructions", {}
            )


def test_provider_errors_do_not_print_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        ResponsesClient()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(401, text="secret-content"))
    ) as client:
        with pytest.raises(ValueError, match="HTTP 401") as caught:
            ResponsesClient("secret-key", client).complete(
                OpenAIConfig(model="example"), "instructions", {}
            )
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "text,valid",
    [
        (' {"correctness":4,"faithfulness":3,"explanation":"Supported by backup policy."}', True),
        ('{"correctness":5,"faithfulness":4,"explanation":"yes"}', False),
        ('{"correctness":true,"faithfulness":4,"explanation":"yes"}', False),
        ("not json", False),
        ('{"correctness":4,"faithfulness":4,"explanation":"yes","extra":1}', False),
    ],
)
def test_judge_strict_rubric(text: str, valid: bool) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"]["additionalProperties"] is False
        assert json.loads(body["input"])["reference_answer"] == "seven days"
        assert "untrusted" in body["instructions"]
        return httpx.Response(200, json=response(text))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        judge = LLMJudge(JudgeConfig(model="example"), ResponsesClient("test", client))
        if not valid:
            with pytest.raises(ValueError, match="invalid rubric"):
                judge.assess("How long?", "seven days", "seven days", [SOURCE])
        else:
            result = judge.assess("How long?", "seven days", "seven days", [SOURCE])
            assert result.scores.faithfulness == 3
            assert result.rubric_version == "ragbench-judge-v1"


def test_answer_metrics_and_extraction() -> None:
    assert score_answer("Seven DAYS!", "seven days", [SOURCE]) == {
        "answer_exact_match": 1.0,
        "answer_token_f1": 1.0,
        "context_token_precision": 1.0,
    }
    assert score_answer("seven seven", "seven days", [])["answer_token_f1"] == 0.5
    assert score_answer("", "", [])["answer_exact_match"] == 0.0
    generator = ExtractiveGenerator()
    assert generator.generate("When do backups expire?", [SOURCE]).text == SOURCE.text
    assert "Insufficient" in generator.generate("unrelated", [SOURCE]).text


def test_runner_local_answers() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/answers.yaml")
    result = evaluate(config)
    assert result.answer_metrics is not None
    assert result.estimated_cost_usd == 0.0
    assert result.input_tokens is None
    assert all(
        q.answer and q.context_chunk_ids and q.generation_ms is not None for q in result.questions
    )
    assert result.judge_metrics is None


def test_runner_judge_totals_and_comparison_identity() -> None:
    from ragbench.evaluation.comparison import Thresholds, compare
    from ragbench.evaluation.models import EvaluationResult

    config = load_config(Path(__file__).resolve().parents[1] / "configs/answers.yaml")
    prices = Pricing(input_per_million=2.0, cached_input_per_million=1.0, output_per_million=8.0)
    judge_config = JudgeConfig(model="example", pricing=prices)
    config = config.model_copy(update={"judge": judge_config})
    payload = response(
        '{"correctness":2,"faithfulness":4,"explanation":"Partial answer, supported."}'
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        report = evaluate(config, judge=LLMJudge(judge_config, ResponsesClient("test", client)))
    assert report.judge_metrics == {"judge_correctness": 0.5, "judge_faithfulness": 1.0}
    assert report.input_tokens == 100 * report.question_count
    assert report.output_tokens == 10 * report.question_count
    assert report.estimated_cost_usd == pytest.approx(0.00026 * report.question_count)
    assert EvaluationResult.model_validate_json(report.model_dump_json()) == report
    limits = Thresholds(max_drop={"judge_correctness": 0.0})
    assert compare(report, report, limits).passed
    changed = tuple(
        q.model_copy(update={"judge": q.judge.model_copy(update={"rubric_version": "v2"})})
        for q in report.questions
    )
    with pytest.raises(ValueError, match="resolved model and rubric"):
        compare(report, report.model_copy(update={"questions": changed}), limits)
    with pytest.raises(ValueError, match="Aggregate judge_metrics"):
        EvaluationResult.model_validate_json(
            report.model_copy(
                update={"judge_metrics": {"judge_correctness": 1.0, "judge_faithfulness": 1.0}}
            ).model_dump_json()
        )


def test_timeout_and_generation_config_validation() -> None:
    from ragbench.config import GenerationConfig, RunConfig

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private provider details", request=request)

    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(ValueError, match="connection error or timeout"):
            ResponsesClient("test", client).complete(
                OpenAIConfig(model="example"), "instructions", {}
            )
    with pytest.raises(ValueError):
        GenerationConfig(provider="openai")
    with pytest.raises(ValueError):
        RunConfig(
            dataset={"documents_path": ".", "benchmark_path": "a.json"},
            judge=JudgeConfig(model="example"),
        )


def test_percentile_interpolation() -> None:
    from ragbench.evaluation.timing import percentile

    assert percentile([10.0, 0.0], 0.95) == 9.5
    assert percentile([3.0], 0.5) == 3.0
