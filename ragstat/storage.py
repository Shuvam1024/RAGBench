"""Transactional SQLite run history; reads never create missing databases."""

import sqlite3
from contextlib import closing
from pathlib import Path

from ragstat.evaluation.models import EvaluationResult

SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, retriever TEXT NOT NULL,
    mrr REAL NOT NULL, report_json TEXT NOT NULL
)"""


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def save(self, report: EvaluationResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise ValueError(f"Unsupported run database version: {version}")
                connection.execute(SCHEMA)
                connection.execute("PRAGMA user_version = 1")
                connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                    (
                        report.run_id,
                        report.created_at.isoformat(),
                        report.config.retrieval.type,
                        report.mrr,
                        report.model_dump_json(),
                    ),
                )
        except sqlite3.Error as exc:
            raise ValueError(f"Cannot save run database {self.path}: {exc}") from exc

    def _read(self, sql: str, parameters: tuple[object, ...]) -> list[tuple]:
        try:
            with closing(
                sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=10)
            ) as connection:
                if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise ValueError("Unsupported run database version")
                return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"Cannot read run database {self.path}: {exc}") from exc

    def list(self, limit: int = 20) -> list[dict[str, str | float]]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        rows = self._read(
            "SELECT run_id, created_at, retriever, mrr FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?",
            (limit,),
        )
        return [
            dict(zip(("run_id", "created_at", "retriever", "mrr"), row, strict=True))
            for row in rows
        ]

    def get(self, run_id: str) -> EvaluationResult:
        rows = self._read("SELECT report_json FROM runs WHERE run_id = ?", (run_id,))
        if not rows:
            raise KeyError(run_id)
        return EvaluationResult.model_validate_json(rows[0][0])
