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
    chunk_size: PositiveInt = 120
    overlap: NonnegativeInt = 20

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class BM25Config(ConfigModel):
    type: Literal["bm25"] = "bm25"
    k1: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = 1.5
    b: UnitFloat = 0.75


class DenseConfig(ConfigModel):
    type: Literal["dense"] = "dense"
    model_name: Nonblank = "sentence-transformers/all-MiniLM-L6-v2"
    revision: Nonblank | None = None
    batch_size: PositiveInt = 32
    device: Literal["cpu"] = "cpu"


class HybridConfig(ConfigModel):
    type: Literal["hybrid"] = "hybrid"
    dense_weight: UnitFloat = 0.5
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)


RetrieverConfig = Annotated[BM25Config | DenseConfig | HybridConfig, Field(discriminator="type")]


class EvaluationConfig(ConfigModel):
    recall_at_k: tuple[PositiveInt, ...] = (1, 3, 5)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> Self:
        if not self.recall_at_k or len(set(self.recall_at_k)) != len(self.recall_at_k):
            raise ValueError("recall_at_k must contain distinct positive cutoffs")
        return self


class OutputConfig(ConfigModel):
    json_path: Path | None = None


class RunConfig(ConfigModel):
    schema_version: SchemaVersion = 1
    dataset: DatasetConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrieverConfig = Field(default_factory=BM25Config)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


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
    dataset = config.dataset.model_copy(update={
        "documents_path": (path.parent / config.dataset.documents_path).resolve(),
        "benchmark_path": (path.parent / config.dataset.benchmark_path).resolve(),
    })
    output = config.output
    if output.json_path is not None:
        output = output.model_copy(update={"json_path": (path.parent / output.json_path).resolve()})
    return config.model_copy(update={"dataset": dataset, "output": output})
