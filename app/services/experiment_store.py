"""SQLite persistence for experiment runs and comparison feedback."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(database_path: str) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_store(database_path: str) -> None:
    """Create the local experiment tables if they do not exist."""
    with closing(_connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS comparison_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                prompt TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS comparison_feedback (
                run_id TEXT PRIMARY KEY REFERENCES comparison_runs(run_id),
                created_at TEXT NOT NULL,
                preferred_path TEXT NOT NULL,
                direct_vlm_rating INTEGER,
                structured_pipeline_rating INTEGER,
                comment TEXT
            )
            """
        )


def save_comparison(database_path: str, result: dict[str, Any]) -> None:
    """Persist a comparison without retaining the uploaded image bytes."""
    initialize_store(database_path)
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO comparison_runs (run_id, created_at, prompt, result_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                result["run_id"],
                created_at,
                result["user_prompt"],
                json.dumps(result, ensure_ascii=False),
            ),
        )


def save_feedback(
    database_path: str,
    *,
    run_id: str,
    preferred_path: str,
    direct_vlm_rating: int | None,
    structured_pipeline_rating: int | None,
    comment: str | None,
) -> None:
    """Persist a user's preference and optional per-answer ratings."""
    initialize_store(database_path)
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect(database_path) as connection:
        run_exists = connection.execute(
            "SELECT 1 FROM comparison_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run_exists:
            raise KeyError(f"Comparison run '{run_id}' was not found.")
        connection.execute(
            """
            INSERT INTO comparison_feedback (
                run_id, created_at, preferred_path, direct_vlm_rating,
                structured_pipeline_rating, comment
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                created_at = excluded.created_at,
                preferred_path = excluded.preferred_path,
                direct_vlm_rating = excluded.direct_vlm_rating,
                structured_pipeline_rating = excluded.structured_pipeline_rating,
                comment = excluded.comment
            """,
            (
                run_id,
                created_at,
                preferred_path,
                direct_vlm_rating,
                structured_pipeline_rating,
                comment,
            ),
        )
