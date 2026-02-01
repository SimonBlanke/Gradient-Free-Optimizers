"""SQLite storage backend."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from ..record import EvaluationRecord, ExperimentMetadata


class SQLiteBackend:
    """
    SQLite-based storage backend.

    Stores evaluation records and metadata in a local SQLite database.
    This is the default backend - lightweight, no external dependencies,
    and supports concurrent reads.

    Args:
        path: Path to the SQLite database file (e.g., "experiment.db")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Create/open the database and ensure tables exist."""
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,  # Allow multi-threaded access
            isolation_level=None,  # Autocommit mode
        )
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the required tables if they don't exist."""
        assert self._conn is not None

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                search_space TEXT,
                description TEXT,
                tags TEXT,
                schema_version INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER NOT NULL,
                parameters TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp TEXT NOT NULL,
                evaluation_time REAL NOT NULL,
                run_id TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_records_run_id ON records(run_id);
            CREATE INDEX IF NOT EXISTS idx_records_iteration ON records(iteration);
            """
        )

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def append(self, record: EvaluationRecord) -> None:
        """Append a single evaluation record."""
        assert self._conn is not None

        self._conn.execute(
            """
            INSERT INTO records
                (iteration, parameters, score, timestamp, evaluation_time, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.iteration,
                json.dumps(record.parameters),
                record.score,
                record.timestamp.isoformat(),
                record.evaluation_time,
                record.run_id,
            ),
        )

    def append_batch(self, records: list[EvaluationRecord]) -> None:
        """Append multiple records in a single transaction (more efficient)."""
        assert self._conn is not None

        self._conn.executemany(
            """
            INSERT INTO records
                (iteration, parameters, score, timestamp, evaluation_time, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.iteration,
                    json.dumps(r.parameters),
                    r.score,
                    r.timestamp.isoformat(),
                    r.evaluation_time,
                    r.run_id,
                )
                for r in records
            ],
        )

    def get_records(self, run_id: str | None = None) -> Iterator[EvaluationRecord]:
        """Retrieve evaluation records, optionally filtered by run_id."""
        assert self._conn is not None

        if run_id is not None:
            cursor = self._conn.execute(
                "SELECT * FROM records WHERE run_id = ? ORDER BY iteration",
                (run_id,),
            )
        else:
            cursor = self._conn.execute("SELECT * FROM records ORDER BY iteration")

        for row in cursor:
            yield EvaluationRecord(
                iteration=row["iteration"],
                parameters=json.loads(row["parameters"]),
                score=row["score"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                evaluation_time=row["evaluation_time"],
                run_id=row["run_id"],
            )

    def save_metadata(self, metadata: ExperimentMetadata) -> None:
        """Save experiment metadata (upsert)."""
        assert self._conn is not None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO metadata
                (id, name, created_at, search_space, description, tags, schema_version)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.name,
                metadata.created_at.isoformat(),
                json.dumps(metadata.search_space) if metadata.search_space else None,
                metadata.description,
                json.dumps(metadata.tags),
                metadata.schema_version,
            ),
        )

    def load_metadata(self) -> ExperimentMetadata | None:
        """Load experiment metadata."""
        assert self._conn is not None

        cursor = self._conn.execute("SELECT * FROM metadata WHERE id = 1")
        row = cursor.fetchone()

        if row is None:
            return None

        return ExperimentMetadata(
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            search_space=json.loads(row["search_space"])
            if row["search_space"]
            else None,
            description=row["description"],
            tags=json.loads(row["tags"]) if row["tags"] else {},
            schema_version=row["schema_version"],
        )

    def list_runs(self) -> list[str]:
        """List all unique run IDs."""
        assert self._conn is not None

        cursor = self._conn.execute(
            "SELECT DISTINCT run_id FROM records ORDER BY run_id"
        )
        return [row["run_id"] for row in cursor]

    def count_records(self, run_id: str | None = None) -> int:
        """Count records, optionally filtered by run_id."""
        assert self._conn is not None

        if run_id is not None:
            cursor = self._conn.execute(
                "SELECT COUNT(*) as count FROM records WHERE run_id = ?",
                (run_id,),
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) as count FROM records")

        return cursor.fetchone()["count"]

    def get_best_record(self, run_id: str | None = None) -> EvaluationRecord | None:
        """Get the record with the highest score."""
        assert self._conn is not None

        if run_id is not None:
            cursor = self._conn.execute(
                "SELECT * FROM records WHERE run_id = ? ORDER BY score DESC LIMIT 1",
                (run_id,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM records ORDER BY score DESC LIMIT 1"
            )

        row = cursor.fetchone()
        if row is None:
            return None

        return EvaluationRecord(
            iteration=row["iteration"],
            parameters=json.loads(row["parameters"]),
            score=row["score"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            evaluation_time=row["evaluation_time"],
            run_id=row["run_id"],
        )

    def __enter__(self) -> SQLiteBackend:
        """Open database connection for context manager use."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close database connection when exiting context."""
        self.disconnect()
