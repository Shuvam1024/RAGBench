"""Read-only API for a local run database; evaluation stays in the CLI."""

import importlib.metadata
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from ragbench.evaluation.models import EvaluationResult
from ragbench.storage import RunStore


def package_version() -> str:
    try:
        return importlib.metadata.version("ragbench")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def create_app(database: Path) -> FastAPI:
    app = FastAPI(
        title="RAGBench",
        version=package_version(),
        description="Read-only evaluation run history",
    )
    store = RunStore(database)

    @app.exception_handler(ValueError)
    async def database_error(request: object, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": "Run database unavailable or incompatible"}
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        store.list(1)
        return {"status": "ok"}

    @app.get("/runs")
    def runs(limit: int = Query(20, ge=1, le=1000)) -> list[dict[str, str | float]]:
        return store.list(limit)

    @app.get("/runs/{run_id}", response_model=EvaluationResult)
    def run(run_id: str) -> EvaluationResult:
        try:
            return store.get(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Run not found") from exc

    return app
