"""Typed experiment settings and YAML paths resolved relative to their file."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
UnitFloat = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Nonblank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
SchemaVersion = Annotated[StrictInt, Field(ge=1, le=1)]


class ConfigModel(BaseModel):
    """Reject misspellings and accidental mutation of an experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(ConfigModel):
    documents_path: Path
    benchmark_path: Path


class ChunkingConfig(ConfigModel):
    # ``words`` is the historical overlapping word-window chunker. ``document``
    # indexes each document as one string. The default keeps existing configs
    # on word windows.
    unit: Literal["words", "document"] = "words"
    chunk_size: PositiveInt = 120
    overlap: NonnegativeInt = 20

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.unit == "words" and self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class BM25Config(ConfigModel):
    type: Literal["bm25"] = "bm25"
    k1: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = 1.5
    b: UnitFloat = 0.75
    # ``whitespace`` is the historical ``\\w+`` tokenizer. ``stem`` also drops
    # English stopwords and applies the Snowball English stemmer. The default
    # keeps existing configs and committed reports on the original tokenizer.
    tokenizer: Literal["whitespace", "stem"] = "whitespace"


class DenseConfig(ConfigModel):
    type: Literal["dense"] = "dense"
    model_name: Nonblank = "sentence-transformers/all-MiniLM-L6-v2"
    revision: Nonblank | None = None
    batch_size: PositiveInt = 32
    device: Literal["cpu"] = "cpu"


class HybridConfig(ConfigModel):
    type: Literal["hybrid"] = "hybrid"
    fusion: Literal["minmax", "rrf"] = "minmax"
    dense_weight: UnitFloat = 0.5
    rrf_k: PositiveInt = 60
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)


RetrieverConfig = Annotated[BM25Config | DenseConfig | HybridConfig, Field(discriminator="type")]


class EvaluationConfig(ConfigModel):
    recall_at_k: tuple[PositiveInt, ...] = (1, 3, 5)
    ndcg_at_k: tuple[PositiveInt, ...] | None = None
    precision_at_k: tuple[PositiveInt, ...] | None = None
    stored_hits: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_cutoffs(self) -> Self:
        for name in ("recall_at_k", "ndcg_at_k", "precision_at_k"):
            values = getattr(self, name)
            if values is None:
                continue
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct positive cutoffs")
        return self

    def ndcg_cutoffs(self) -> tuple[int, ...]:
        return self.ndcg_at_k if self.ndcg_at_k is not None else self.recall_at_k

    def precision_cutoffs(self) -> tuple[int, ...]:
        return self.precision_at_k if self.precision_at_k is not None else self.recall_at_k


class OutputConfig(ConfigModel):
    json_path: Path | None = None


NonnegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class Pricing(ConfigModel):
    """User-supplied USD per million tokens; never inferred from a model name."""

    input_per_million: NonnegativeFloat
    cached_input_per_million: NonnegativeFloat
    output_per_million: NonnegativeFloat


class OpenAIConfig(ConfigModel):
    model: Nonblank
    max_output_tokens: PositiveInt = 512
    timeout_seconds: Annotated[float, Field(strict=True, gt=0, le=300)] = 60.0
    pricing: Pricing | None = None


class GenerationConfig(ConfigModel):
    provider: Literal["extractive", "openai"] = "extractive"
    context_k: PositiveInt = 3
    max_context_chars: PositiveInt = 16000
    openai: OpenAIConfig | None = None

    @model_validator(mode="after")
    def provider_settings(self) -> Self:
        if (self.provider == "openai") != (self.openai is not None):
            raise ValueError("openai settings are required only for provider=openai")
        return self


class JudgeConfig(OpenAIConfig):
    max_output_tokens: PositiveInt = 1024


class StorageConfig(ConfigModel):
    sqlite_path: Path | None = None


class RunConfig(ConfigModel):
    schema_version: SchemaVersion = 1
    dataset: DatasetConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrieverConfig = Field(default_factory=BM25Config)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    generation: GenerationConfig | None = None
    judge: JudgeConfig | None = None
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def judge_needs_answers(self) -> Self:
        if self.judge is not None and self.generation is None:
            raise ValueError("judge requires generation to be configured")
        return self


def load_config(path: Path) -> RunConfig:
    """Load a safe YAML mapping and resolve its paths independently of cwd."""
    path = path.resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration {path} must be a YAML mapping")
    config = RunConfig.model_validate(raw)
    dataset = config.dataset.model_copy(
        update={
            "documents_path": (path.parent / config.dataset.documents_path).resolve(),
            "benchmark_path": (path.parent / config.dataset.benchmark_path).resolve(),
        }
    )
    output = config.output
    if output.json_path is not None:
        output = output.model_copy(update={"json_path": (path.parent / output.json_path).resolve()})
    storage = config.storage
    if storage.sqlite_path is not None:
        storage = storage.model_copy(
            update={"sqlite_path": (path.parent / storage.sqlite_path).resolve()}
        )
    return config.model_copy(update={"dataset": dataset, "output": output, "storage": storage})
