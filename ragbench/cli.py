"""Typer entry point: load settings, run evaluation, report, optionally save JSON."""

import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from ragbench.config import RunConfig, load_config
from ragbench.evaluation.comparison import compare as compare_reports
from ragbench.evaluation.comparison import load_report, load_thresholds
from ragbench.evaluation.runner import evaluate as run_evaluation
from ragbench.storage import RunStore

app = typer.Typer(
    no_args_is_help=True, help="Evaluate retrieval configurations on a labeled corpus."
)


@app.callback()
def main() -> None:
    """RAGBench: reproducible retrieval evaluation."""


def validate_output(path: Path, config_path: Path, config: RunConfig) -> None:
    """Keep output outside the input corpus; protect aliases and hard links too."""
    protected = [config_path.resolve(), config.dataset.benchmark_path.resolve()]
    root = config.dataset.documents_path.resolve()
    if path == root or root in path.parents:
        raise ValueError("Output must be outside the documents directory")
    if root.is_dir():
        protected.extend(item for item in root.rglob("*") if item.is_file())
    for source in protected:
        if path == source or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"Output would overwrite an input: {source}")
    if path.exists() and path.is_dir():
        raise ValueError(f"Output is a directory: {path}")


def save_result(result: BaseModel, path: Path) -> None:
    """Replace reports atomically so interrupted writes cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".ragbench-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(result.model_dump_json(indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="YAML experiment configuration")],
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional JSON output path")
    ] = None,
    db: Annotated[Path | None, typer.Option("--db", help="Optional SQLite run history")] = None,
) -> None:
    """Build an index and evaluate all benchmark questions."""
    try:
        settings = load_config(config)
        destination = output.resolve() if output is not None else settings.output.json_path
        database = db.resolve() if db is not None else settings.storage.sqlite_path
        if destination is not None:
            validate_output(destination, config, settings)
        if database is not None:
            validate_output(database, config, settings)
            if destination == database or (
                destination is not None
                and destination.exists()
                and database.exists()
                and destination.samefile(database)
            ):
                raise ValueError("JSON output and SQLite database must be different files")
        settings = settings.model_copy(
            update={"output": settings.output.model_copy(update={"json_path": destination})}
        )
        settings = settings.model_copy(
            update={"storage": settings.storage.model_copy(update={"sqlite_path": database})}
        )
        result = run_evaluation(settings)
        if database is not None:
            RunStore(database).save(result)
        if destination is not None:
            save_result(result, destination)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"RAGBench | {settings.retrieval.type}")
    typer.echo(
        f"Documents: {result.document_count} | Chunks: {result.chunk_count} | Questions: {result.question_count}"
    )
    if result.skipped_empty_files:
        typer.echo(f"Skipped empty files: {result.skipped_empty_files}")
    for k, value in result.recall_at_k.items():
        typer.echo(f"Recall@{k:<3} {value:.4f}")
    typer.echo(f"MRR        {result.mrr:.4f} (full document ranking)")
    typer.echo(f"Retrieval p95: {result.timings.retrieval_p95_ms:.3f} ms | Run: {result.run_id}")
    for name, value in (result.answer_metrics or {}).items():
        typer.echo(f"{name}: {value:.4f}")
    for name, value in (result.judge_metrics or {}).items():
        typer.echo(f"{name}: {value:.4f} (LLM assessment)")
    if result.input_tokens is not None:
        typer.echo(
            f"Tokens: {result.input_tokens} input / {result.output_tokens} output (generation + judge)"
        )
    if result.estimated_cost_usd is not None:
        typer.echo(f"Estimated API cost: ${result.estimated_cost_usd:.6f}")
    if destination is not None:
        typer.echo(f"Saved: {destination}")


@app.command("compare")
def compare_command(
    baseline: Annotated[Path, typer.Option("--baseline")],
    candidate: Annotated[Path, typer.Option("--candidate")],
    thresholds: Annotated[Path, typer.Option("--thresholds")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Compare compatible reports; exit 2 when a regression exceeds tolerance."""
    try:
        if output is not None:
            destination = output.resolve()
            for source in (baseline, candidate, thresholds):
                if destination == source.resolve() or (
                    destination.exists() and destination.samefile(source)
                ):
                    raise ValueError("Comparison output would overwrite an input")
        result = compare_reports(
            load_report(baseline), load_report(candidate), load_thresholds(thresholds)
        )
        if output is not None:
            save_result(result, output.resolve())
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for check in result.checks:
        typer.echo(
            f"{'PASS' if check.passed else 'FAIL'} {check.metric}: {check.baseline:.6f} -> {check.candidate:.6f}"
        )
    typer.echo("Regression check passed" if result.passed else "Regression detected")
    if not result.passed:
        raise typer.Exit(2)


@app.command("history")
def history_command(db: Annotated[Path, typer.Option("--db")], limit: int = 20) -> None:
    """List saved runs without opening an API server."""
    try:
        for row in RunStore(db).list(limit):
            typer.echo(
                f"{row['run_id']}  {row['created_at']}  {row['retriever']}  MRR={row['mrr']:.4f}"
            )
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("serve")
def serve_command(
    db: Annotated[Path, typer.Option("--db")], host: str = "127.0.0.1", port: int = 8000
) -> None:
    """Serve a local, read-only run-history API (requires the api extra)."""
    try:
        import uvicorn

        from ragbench.api import create_app

        RunStore(db).list(1)
        uvicorn.run(create_app(db), host=host, port=port)
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"Error: {exc}. Install the api extra if needed.", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
