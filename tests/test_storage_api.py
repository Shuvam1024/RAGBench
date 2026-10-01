from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config import load_config
from ragbench.evaluation.models import EvaluationResult
from ragbench.evaluation.runner import evaluate
from ragbench.storage import RunStore


@pytest.fixture
def report() -> EvaluationResult:
    return evaluate(load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml"))


def test_store_roundtrip_and_duplicate_atomicity(tmp_path: Path, report: EvaluationResult) -> None:
    store = RunStore(tmp_path / "runs.sqlite")
    store.save(report)
    assert store.get(report.run_id) == report
    with pytest.raises(ValueError, match="Cannot save"):
        store.save(report)
    assert len(store.list()) == 1
    with pytest.raises(KeyError):
        store.get("' OR 1=1 --")
    with pytest.raises(ValueError):
        store.list(0)


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ValueError):
        RunStore(path).list()
    assert not path.exists()


def test_read_only_api(tmp_path: Path, report: EvaluationResult) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ragbench.api import create_app

    path = tmp_path / "runs.sqlite"
    client = TestClient(create_app(path))
    assert client.get("/health").status_code == 503
    assert not path.exists()
    RunStore(path).save(report)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/runs").json()[0]["run_id"] == report.run_id
    assert client.get(f"/runs/{report.run_id}").json()["question_count"] == report.question_count
    assert client.get("/runs/missing").status_code == 404
    assert client.get("/runs?limit=0").status_code == 422
    assert client.post("/runs").status_code == 405


def test_cli_database_and_collision(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"
    database = tmp_path / "runs.sqlite"
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "--config", str(config), "--db", str(database)])
    assert result.exit_code == 0, result.output
    assert "bm25" in runner.invoke(app, ["history", "--db", str(database)]).output
    result = runner.invoke(
        app, ["evaluate", "--config", str(config), "--db", str(database), "--output", str(database)]
    )
    assert result.exit_code == 1
    assert len(RunStore(database).list()) == 1
