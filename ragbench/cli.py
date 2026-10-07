"""Typer entry point: load settings, run evaluation, report, optionally save JSON."""

import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from ragbench.config import RunConfig, load_config
from ragbench.datasets.beir import DATASETS, prepare_dataset
from ragbench.evaluation.comparison import compare as compare_reports
from ragbench.evaluation.comparison import load_report, load_thresholds
from ragbench.evaluation.runner import evaluate as run_evaluation
from ragbench.evaluation.runner import relativize_result
from ragbench.evaluation.sweep import load_sweep, run_sweep
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
    portable: Annotated[
        bool,
        typer.Option("--portable", help="Store paths relative to the working directory"),
    ] = False,
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
        if portable:
            result = relativize_result(result, Path.cwd())
        if database is not None:
            RunStore(database).save(result)
        if destination is not None:
            save_result(result, destination)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"RAGBench | {settings.retrieval.type}")
    typer.echo(
        f"Documents: {result.document_count} | Chunks: {result.chunk_count} | "
        f"Questions: {result.question_count}"
    )
    if result.multi_chunk_documents is not None:
        typer.echo(f"Multi-chunk documents: {result.multi_chunk_documents}")
    if result.skipped_empty_files:
        typer.echo(f"Skipped empty files: {result.skipped_empty_files}")
    for k, value in result.recall_at_k.items():
        typer.echo(f"Recall@{k:<3} {value:.4f}")
    for k, value in (result.ndcg_at_k or {}).items():
        typer.echo(f"nDCG@{k:<5} {value:.4f}")
    for k, value in (result.precision_at_k or {}).items():
        typer.echo(f"P@{k:<8} {value:.4f}")
    if result.mean_average_precision is not None:
        typer.echo(f"MAP        {result.mean_average_precision:.4f}")
    if result.candidate_recall_at_100 is not None:
        typer.echo(f"MRR        {result.mrr:.4f} (reranked candidate set)")
    else:
        typer.echo(f"MRR        {result.mrr:.4f} (full document ranking)")
    if result.candidate_recall_at_100 is not None:
        typer.echo(
            "Candidate Recall@100 "
            f"{result.candidate_recall_at_100:.4f} (first stage, before rerank)"
        )
    typer.echo(
        f"Full-ranking retrieval p95: {result.timings.retrieval_p95_ms:.3f} ms | "
        f"Run: {result.run_id}"
    )
    if result.timings.rerank_p95_ms is not None:
        typer.echo(f"Rerank p95: {result.timings.rerank_p95_ms:.3f} ms")
    if result.timings.pipeline_p95_ms is not None:
        typer.echo(
            f"Top-K pipeline p95: {result.timings.pipeline_p95_ms:.3f} ms "
            "(full-ranking first stage plus rerank)"
        )
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
        interval = ""
        if check.ci_low is not None and check.ci_high is not None and check.mean_delta is not None:
            interval = (
                f" | delta {check.mean_delta:+.6f} CI [{check.ci_low:+.6f}, {check.ci_high:+.6f}]"
            )
            if check.permutation_p_value is not None:
                interval += f" p={check.permutation_p_value:.4f}"
        typer.echo(
            f"{'PASS' if check.passed else 'FAIL'} {check.metric}: "
            f"{check.baseline:.6f} -> {check.candidate:.6f}{interval}"
        )
        changed = [
            item
            for item in result.question_changes
            if item.metric == check.metric and item.direction != "unchanged"
        ]
        improved = [item.question_id for item in changed if item.direction == "improved"]
        regressed = [item.question_id for item in changed if item.direction == "regressed"]
        typer.echo(
            f"  improved {len(improved)} | regressed {len(regressed)} | "
            f"unchanged {sum(item.metric == check.metric for item in result.question_changes) - len(changed)}"
        )
        if regressed:
            typer.echo("  regressed questions: " + ", ".join(regressed[:20]))
        if improved:
            typer.echo("  improved questions: " + ", ".join(improved[:20]))
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


@app.command("dataset")
def dataset_command(
    name: Annotated[str, typer.Argument(help="Public dataset name, such as scifact")],
    cache: Annotated[Path, typer.Option("--cache")] = Path(".cache/beir"),
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="BEIR qrels split to materialize. Defaults to the dataset's test split.",
        ),
    ] = None,
) -> None:
    """Download a checksum-verified corpus and write documents.jsonl plus a benchmark."""
    try:
        result = prepare_dataset(name, cache, manifest_path=manifest, force=force, split=split)
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Prepared {result.dataset} {result.split}: {result.document_count} documents, "
        f"{result.query_count} questions, sha256 {result.sha256}"
    )
    typer.echo(
        f"Words: min {result.word_length.minimum} | p50 {result.word_length.p50:.1f} | "
        f"mean {result.word_length.mean:.1f} | max {result.word_length.maximum}"
    )
    known = ", ".join(sorted(DATASETS))
    typer.echo(f"Cache: {(cache / name).resolve()} | known datasets: {known}")


@app.command("sweep")
def sweep_command(
    config: Annotated[Path, typer.Option("--config", help="YAML one-factor sweep")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Re-run a base configuration, changing one field at a time."""
    try:
        _base_path, sweep, base = load_sweep(config)
        result = run_sweep(base, sweep, sweep.base)
        if output is not None:
            destination = output.resolve()
            if destination == config.resolve() or (
                destination.exists() and destination.samefile(config)
            ):
                raise ValueError("Sweep output would overwrite the sweep configuration")
            save_result(result, destination)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for row in result.rows:
        mrr = row.metrics.get("mrr", float("nan"))
        ndcg = row.metrics.get("ndcg@10")
        extra = f" nDCG@10={ndcg:.4f}" if ndcg is not None else ""
        typer.echo(
            f"{row.name}: chunks={row.chunk_count} multi={row.multi_chunk_documents} "
            f"MRR={mrr:.4f}{extra}"
        )
    if output is not None:
        typer.echo(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    app()
