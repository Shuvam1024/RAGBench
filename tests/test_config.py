from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.config import (
    BM25Config,
    ChunkingConfig,
    DenseConfig,
    EvaluationConfig,
    HybridConfig,
    RunConfig,
    load_config,
)


@pytest.mark.parametrize(
    "settings",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": True},
        {"chunk_size": "30"},
        {"chunk_size": 5, "overlap": 5},
        {"overlap": -1},
        {"overlap": 1.5},
        {"overlap": False},
        {"overlapp": 1},
    ],
)
def test_invalid_chunking(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig.model_validate(settings)


@pytest.mark.parametrize("cutoffs", [[], [0], [1, 1], [True], ["1"]])
def test_invalid_cutoffs(cutoffs: list[object]) -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig.model_validate({"recall_at_k": cutoffs})


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_invalid_hybrid_weight(weight: object) -> None:
    with pytest.raises(ValidationError):
        HybridConfig.model_validate({"dense_weight": weight})


def test_config_paths_follow_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "configs" / "run.yaml"
    config.parent.mkdir()
    config.write_text(
        "dataset:\n  documents_path: ../docs\n  benchmark_path: ../questions.json\n"
        "output:\n  json_path: ../output.json\nretrieval:\n  type: dense\n"
    )
    monkeypatch.chdir(tmp_path.parent)
    parsed = load_config(config)
    assert parsed.dataset.documents_path == tmp_path / "docs"
    assert parsed.dataset.benchmark_path == tmp_path / "questions.json"
    assert parsed.output.json_path == tmp_path / "output.json"
    assert isinstance(parsed.retrieval, DenseConfig)


@pytest.mark.parametrize("content", ["[", "[]", "null", "!!python/object:bad {}", "dataset: {}"])
def test_bad_yaml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "extra",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"typo": 1},
        {"retrieval": {"type": "unknown"}},
        {"retrieval": {"type": "bm25", "b": 2}},
        {"retrieval": {"type": "bm25", "k1": 0}},
        {"retrieval": {"type": "bm25", "tokenizer": "porter"}},
    ],
)
def test_bad_root_config(extra: dict[str, object]) -> None:
    raw = {"dataset": {"documents_path": "docs", "benchmark_path": "questions.json"}, **extra}
    with pytest.raises(ValueError):
        RunConfig.model_validate(raw)


def test_bm25_tokenizer_defaults_to_whitespace_for_older_configs() -> None:
    parsed = BM25Config.model_validate({"type": "bm25", "k1": 1.5, "b": 0.75})
    assert parsed.tokenizer == "whitespace"
    hybrid = HybridConfig.model_validate({"bm25": {"k1": 1.2}})
    assert hybrid.bm25.tokenizer == "whitespace"


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot read configuration"):
        load_config(tmp_path / "missing.yaml")
