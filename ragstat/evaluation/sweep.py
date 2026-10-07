"""One-factor-at-a-time sweeps over a validated base configuration.

Each row starts from the base run and changes a single field. The base value of
an axis is not repeated. Dense embedding vectors are reused by the embedder
cache when the chunk text is unchanged, so hybrid-weight rows do not re-encode
the corpus.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from ragstat.config import ConfigModel, Nonblank, RunConfig, SchemaVersion, load_config
from ragstat.evaluation.models import EvaluationResult, Record
from ragstat.evaluation.runner import evaluate


class SweepAxis(ConfigModel):
    name: Nonblank
    field: Nonblank
    values: tuple[int | float | str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_booleans(self) -> Self:
        if any(isinstance(value, bool) for value in self.values):
            raise ValueError("Sweep values must not be booleans")
        return self


class SweepConfig(ConfigModel):
    schema_version: SchemaVersion = 1
    base: Nonblank
    axes: tuple[SweepAxis, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_axes(self) -> Self:
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("Sweep axis names must be unique")
        return self


class SweepRow(Record):
    name: str
    axis: str
    field: str
    value: str
    document_count: int
    chunk_count: int
    multi_chunk_documents: int
    question_count: int
    metrics: dict[str, float]
    index_ms: float
    retrieval_p50_ms: float
    retriever: str
    fusion: str | None = None


class SweepResult(Record):
    schema_version: int = 1
    base_config: str
    corpus_sha256: str
    benchmark_sha256: str
    rows: tuple[SweepRow, ...]


def load_sweep(path: Path) -> tuple[Path, SweepConfig, RunConfig]:
    path = path.resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read sweep {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Sweep {path} must be a YAML mapping")
    sweep = SweepConfig.model_validate(raw)
    base_path = (path.parent / sweep.base).resolve()
    return base_path, sweep, load_config(base_path)


def _split_field(field: str) -> tuple[str, str]:
    section, separator, name = field.partition(".")
    if not separator or not section or not name or "." in name:
        raise ValueError(f"Sweep field must be one dotted section.name path: {field}")
    return section, name


def read_field(config: RunConfig, field: str) -> object:
    section, name = _split_field(field)
    holder = getattr(config, section, None)
    if holder is None or not hasattr(holder, name):
        raise ValueError(f"Unknown sweep field: {field}")
    return getattr(holder, name)


def with_field(config: RunConfig, field: str, value: object) -> RunConfig:
    section, name = _split_field(field)
    data = config.model_dump(mode="python")
    current = data.get(section)
    if not isinstance(current, dict) or name not in current:
        raise ValueError(f"Unknown sweep field: {field}")
    current[name] = value
    try:
        return RunConfig.model_validate(data)
    except ValueError as exc:
        raise ValueError(f"Invalid sweep value {value!r} for {field}") from exc


def _display(value: object) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _metrics(result: EvaluationResult) -> dict[str, float]:
    metrics: dict[str, float] = {"mrr": result.mrr}
    if result.mean_average_precision is not None:
        metrics["map"] = result.mean_average_precision
    for cutoff, value in result.recall_at_k.items():
        metrics[f"recall@{cutoff}"] = value
    for cutoff, value in (result.ndcg_at_k or {}).items():
        metrics[f"ndcg@{cutoff}"] = value
    for cutoff, value in (result.precision_at_k or {}).items():
        metrics[f"precision@{cutoff}"] = value
    return metrics


def _row(
    name: str, axis: str, field: str, value: str, result: EvaluationResult, config: RunConfig
) -> SweepRow:
    if result.multi_chunk_documents is None:
        raise ValueError("Sweep rows require multi_chunk_documents")
    fusion = result.retriever.get("fusion")
    return SweepRow(
        name=name,
        axis=axis,
        field=field,
        value=value,
        document_count=result.document_count,
        chunk_count=result.chunk_count,
        multi_chunk_documents=result.multi_chunk_documents,
        question_count=result.question_count,
        metrics=_metrics(result),
        index_ms=result.timings.index_ms,
        retrieval_p50_ms=result.timings.retrieval_p50_ms,
        retriever=config.retrieval.type,
        fusion=str(fusion) if fusion is not None else None,
    )


def run_sweep(base: RunConfig, sweep: SweepConfig, base_name: str) -> SweepResult:
    base_result = evaluate(base)
    rows = [_row("base", "base", "", "base", base_result, base)]
    for axis in sweep.axes:
        current = read_field(base, axis.field)
        for value in axis.values:
            if value == current:
                continue
            updated = with_field(base, axis.field, value)
            result = evaluate(updated)
            shown = _display(value)
            rows.append(_row(f"{axis.name}={shown}", axis.name, axis.field, shown, result, updated))
    return SweepResult(
        base_config=base_name,
        corpus_sha256=base_result.corpus_sha256,
        benchmark_sha256=base_result.benchmark_sha256,
        rows=tuple(rows),
    )
