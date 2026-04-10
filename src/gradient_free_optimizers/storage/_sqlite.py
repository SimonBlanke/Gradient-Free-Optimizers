"""SQLite-based persistent storage backend.

Stores evaluation results in a local SQLite database with WAL journaling
for concurrent read access. Suitable for single-machine persistence and
multi-process sharing (e.g. multiple workers on the same host).

Not recommended for network filesystems (NFS, SMB). SQLite's file locking
does not work reliably over network mounts and can cause silent corruption.
For multi-machine setups, use a network-capable storage backend instead.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from typing import TYPE_CHECKING

from ._base import BaseStorage

if TYPE_CHECKING:
    from .._result import Result


class SQLiteStorage(BaseStorage):
    """Persistent evaluation cache backed by a SQLite database.

    Creates (or opens) a SQLite database at the given path and stores
    evaluation results in a single ``evaluations`` table. Uses WAL
    journal mode for concurrent read access from multiple processes.

    Position keys are stored as JSON-encoded strings (arrays of integers).
    Metrics dicts are stored as JSON text.

    Parameters
    ----------
    path : str
        Filesystem path to the SQLite database file.
        Created if it does not exist. Use ``":memory:"`` for a
        transient in-memory database (mostly useful for testing).

    Examples
    --------
    Persistent cache across runs::

        storage = SQLiteStorage("my_optimization.db")
        opt.search(objective, n_iter=100, memory=storage)

        # Later, continue from where you left off:
        opt2 = HillClimbingOptimizer(search_space)
        opt2.search(objective, n_iter=100, memory=storage)

    Multi-process on the same machine::

        storage = SQLiteStorage("/shared/path/results.db")
        # Multiple processes can read/write concurrently via WAL mode
    """

    def __init__(self, path: str):
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS evaluations (
                position_key TEXT PRIMARY KEY,
                score REAL NOT NULL,
                metrics TEXT NOT NULL DEFAULT '{}',
                objectives TEXT DEFAULT NULL
            )"""
        )
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self):
        """Add columns introduced after the initial schema."""
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(evaluations)")
        }
        if "objectives" not in columns:
            self._conn.execute(
                "ALTER TABLE evaluations ADD COLUMN objectives TEXT DEFAULT NULL"
            )
            self._conn.commit()

    def _key_to_str(self, key: tuple) -> str:
        # Convert numpy integers to Python ints for JSON serialization
        return json.dumps([int(k) for k in key])

    def _str_to_key(self, s: str) -> tuple:
        return tuple(json.loads(s))

    def get(self, key: tuple) -> Result | None:
        from .._result import Result

        row = self._conn.execute(
            "SELECT score, metrics, objectives FROM evaluations "
            "WHERE position_key = ?",
            (self._key_to_str(key),),
        ).fetchone()
        if row is None:
            return None
        objectives = json.loads(row[2]) if row[2] else None
        return Result(score=row[0], metrics=json.loads(row[1]), objectives=objectives)

    def put(self, key: tuple, result: Result) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO evaluations "
            "(position_key, score, metrics, objectives) VALUES (?, ?, ?, ?)",
            (
                self._key_to_str(key),
                result.score,
                json.dumps(result.metrics) if result.metrics else "{}",
                json.dumps(result.objectives) if result.objectives else None,
            ),
        )
        self._conn.commit()

    def contains(self, key: tuple) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM evaluations WHERE position_key = ?",
            (self._key_to_str(key),),
        ).fetchone()
        return row is not None

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()
        return row[0]

    def items(self) -> Iterator[tuple[tuple, Result]]:
        """Iterate over all stored evaluations using a server-side cursor.

        Rows are fetched in batches to avoid loading the entire database
        into memory at once.
        """
        from .._result import Result

        cursor = self._conn.execute(
            "SELECT position_key, score, metrics, objectives FROM evaluations"
        )
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for pos_str, score, metrics_str, objectives_str in rows:
                objectives = json.loads(objectives_str) if objectives_str else None
                yield (
                    self._str_to_key(pos_str),
                    Result(
                        score=score,
                        metrics=json.loads(metrics_str),
                        objectives=objectives,
                    ),
                )

    def update(self, mapping: dict) -> None:
        """Bulk-insert via a single transaction for performance."""
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO evaluations "
                "(position_key, score, metrics, objectives) VALUES (?, ?, ?, ?)",
                [
                    (
                        self._key_to_str(key),
                        result.score,
                        json.dumps(result.metrics) if result.metrics else "{}",
                        json.dumps(result.objectives) if result.objectives else None,
                    )
                    for key, result in mapping.items()
                ],
            )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self):
        self._conn.close()
