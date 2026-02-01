"""SearchTracker - the main tracking class."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .record import EvaluationRecord, ExperimentMetadata
from .storage.sqlite import SQLiteBackend

if TYPE_CHECKING:
    import pandas as pd

    from .plotting.accessor import PlotAccessor
    from .storage.protocol import StorageBackend


class SearchTracker:
    """
    Tracks optimization experiments with minimal code invasion.

    The SearchTracker collects data from optimization runs using a decorator
    on the objective function. Data is stored in a SQLite database and can
    be visualized using built-in plotting methods.

    Example:
        tracker = SearchTracker("my_experiment.db")

        @tracker.track
        def objective(x, y):
            return -(x**2 + y**2)

        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=100)

        # Analyze results
        print(tracker.best_parameters)
        tracker.plot.convergence()

    Args:
        database: Path to the SQLite database file (will be created if not exists)
        name: Optional experiment name (defaults to database filename)
        search_space: Optional search space definition (for documentation)
        flush_interval: Number of evaluations before writing to database
    """

    def __init__(
        self,
        database: str | Path,
        name: str | None = None,
        search_space: dict[str, Any] | None = None,
        flush_interval: int = 1,
    ):
        self._database_path = Path(database)
        self._name = name or self._database_path.stem
        self._search_space = search_space
        self._flush_interval = flush_interval

        # Initialize storage
        self._storage: StorageBackend = SQLiteBackend(self._database_path)
        self._storage.connect()

        # In-memory cache for fast access
        self._records: list[EvaluationRecord] = []
        self._buffer: list[EvaluationRecord] = []

        # State
        self._iteration = 0
        self._current_run_id: str = "default"
        self._is_loaded = False

        # Load existing data if database exists
        if self._database_path.exists():
            self._load_from_storage()

        # Save metadata
        self._save_metadata()

    def _load_from_storage(self) -> None:
        """Load existing records from storage into memory."""
        self._records = list(self._storage.get_records())
        if self._records:
            self._iteration = max(r.iteration for r in self._records) + 1
            runs = self._storage.list_runs()
            if runs:
                self._current_run_id = runs[-1]
        self._is_loaded = True

    def _save_metadata(self) -> None:
        """Save experiment metadata to storage."""
        metadata = ExperimentMetadata(
            name=self._name,
            created_at=datetime.now(),
            search_space=self._search_space,
        )
        self._storage.save_metadata(metadata)

    # ─────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def database_path(self) -> Path:
        """Path to the database file."""
        return self._database_path

    @property
    def records(self) -> list[EvaluationRecord]:
        """All evaluation records (read-only copy)."""
        return list(self._records)

    @property
    def n_evaluations(self) -> int:
        """Total number of evaluations."""
        return len(self._records)

    @property
    def best_score(self) -> float:
        """Best (highest) score found."""
        if not self._records:
            return float("-inf")
        return max(r.score for r in self._records)

    @property
    def best_parameters(self) -> dict[str, Any]:
        """Parameters that achieved the best score."""
        if not self._records:
            return {}
        best_record = max(self._records, key=lambda r: r.score)
        return best_record.parameters

    @property
    def best_record(self) -> EvaluationRecord | None:
        """The evaluation record with the best score."""
        if not self._records:
            return None
        return max(self._records, key=lambda r: r.score)

    @property
    def run_ids(self) -> list[str]:
        """List of all run IDs."""
        return list({r.run_id for r in self._records})

    # ─────────────────────────────────────────────────────────
    # Decorator API
    # ─────────────────────────────────────────────────────────

    def track(self, func: Callable) -> Callable:
        """
        Decorate objective function to track calls.

        Automatically captures parameters, scores, and timing information
        for every call to the decorated function.

        Works with both GFO-style objective functions (single dict argument)
        and standard Python functions with named parameters.

        Example (GFO style):
            @tracker.track
            def objective(params):
                return -(params["x"]**2 + params["y"]**2)

        Example (named parameters):
            @tracker.track
            def objective(x, y):
                return -(x**2 + y**2)

        Args:
            func: The objective function to track

        Returns
        -------
            Wrapped function that logs every call
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()

            # Call the actual function
            score = func(*args, **kwargs)

            end_time = datetime.now()
            eval_time = (end_time - start_time).total_seconds()

            # Extract parameters
            # GFO calls objective(params_dict) with a single dict argument
            # Normal Python calls objective(x=1, y=2) with kwargs
            parameters = self._extract_parameters(func, args, kwargs)

            # Create and store record
            record = EvaluationRecord(
                iteration=self._iteration,
                parameters=parameters,
                score=float(score),
                timestamp=end_time,
                evaluation_time=eval_time,
                run_id=self._current_run_id,
            )

            self._add_record(record)
            self._iteration += 1

            return score

        return wrapper

    def _extract_parameters(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> dict[str, Any]:
        """
        Extract parameter dictionary from function arguments.

        Handles two calling conventions:
        1. GFO-style: objective(params_dict) - single dict as first argument
        2. Python-style: objective(x=1, y=2) - named keyword arguments
        """
        # GFO-style: Single dict argument
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return dict(args[0])  # Return a copy

        # Python-style: Extract from args and kwargs
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        parameters = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                parameters[param_names[i]] = arg
        parameters.update(kwargs)

        return parameters

    def _add_record(self, record: EvaluationRecord) -> None:
        """Add a record to the buffer and flush if needed."""
        self._records.append(record)
        self._buffer.append(record)

        if len(self._buffer) >= self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        """Write buffered records to storage."""
        if not self._buffer:
            return

        if hasattr(self._storage, "append_batch"):
            self._storage.append_batch(self._buffer)
        else:
            for record in self._buffer:
                self._storage.append(record)

        self._buffer.clear()

    # ─────────────────────────────────────────────────────────
    # Run Management
    # ─────────────────────────────────────────────────────────

    def start_run(self, name: str | None = None) -> str:
        """
        Start a new run within this experiment.

        Useful for comparing different optimizers or configurations
        within the same experiment.

        Args:
            name: Optional run name (auto-generated if None)

        Returns
        -------
            The run ID

        Example:
            tracker.start_run("hill_climbing")
            opt1.search(objective, n_iter=100)
            tracker.end_run()

            tracker.start_run("random_search")
            opt2.search(objective, n_iter=100)
            tracker.end_run()
        """
        self._flush()
        self._current_run_id = name or str(uuid.uuid4())[:8]
        self._iteration = 0
        return self._current_run_id

    def end_run(self) -> None:
        """End the current run and flush all data."""
        self._flush()

    # ─────────────────────────────────────────────────────────
    # Data Access
    # ─────────────────────────────────────────────────────────

    def get_records(self, run_id: str | None = None) -> list[EvaluationRecord]:
        """
        Get evaluation records, optionally filtered by run ID.

        Args:
            run_id: If provided, only return records for this run

        Returns
        -------
            List of evaluation records
        """
        if run_id is None:
            return list(self._records)
        return [r for r in self._records if r.run_id == run_id]

    def to_dataframe(self, run_id: str | None = None) -> pd.DataFrame:
        """
        Convert records to a pandas DataFrame.

        Args:
            run_id: If provided, only include records for this run

        Returns
        -------
            DataFrame with columns: iteration, score, timestamp,
            evaluation_time, run_id, and all parameter columns
        """
        import pandas as pd

        records = self.get_records(run_id)

        if not records:
            return pd.DataFrame()

        data = []
        for r in records:
            row = {
                "iteration": r.iteration,
                "score": r.score,
                "timestamp": r.timestamp,
                "evaluation_time": r.evaluation_time,
                "run_id": r.run_id,
                **r.parameters,
            }
            data.append(row)

        return pd.DataFrame(data)

    # ─────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Generate a text summary of the experiment.

        Returns
        -------
            Formatted string with experiment statistics
        """
        if not self._records:
            return f"Experiment '{self._name}': No evaluations recorded."

        scores = [r.score for r in self._records]
        times = [r.evaluation_time for r in self._records]
        runs = self.run_ids

        best = self.best_record
        assert best is not None

        lines = [
            f"Experiment: {self._name}",
            "=" * 50,
            f"Database: {self._database_path}",
            f"Runs: {len(runs)} ({', '.join(runs)})",
            f"Evaluations: {len(self._records)}",
            "",
            "Scores:",
            f"  Best:  {max(scores):.6f}",
            f"  Worst: {min(scores):.6f}",
            f"  Mean:  {sum(scores) / len(scores):.6f}",
            "",
            "Best Parameters:",
        ]

        for key, value in best.parameters.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")

        lines.extend(
            [
                "",
                "Timing:",
                f"  Total: {sum(times):.2f}s",
                f"  Mean:  {sum(times) / len(times) * 1000:.2f}ms per evaluation",
                "=" * 50,
            ]
        )

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────────────────────

    @property
    def plot(self) -> PlotAccessor:
        """
        Access plotting methods.

        Returns a PlotAccessor object with methods for various visualizations.

        Example:
            tracker.plot.convergence()
            tracker.plot.search_space(dimensions=["x", "y"])
            tracker.plot.parameter_importance()
        """
        from .plotting.accessor import PlotAccessor

        return PlotAccessor(self)

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the tracker and flush all pending data."""
        self._flush()
        self._storage.disconnect()

    @classmethod
    def load(cls, database: str | Path) -> SearchTracker:
        """
        Load an existing experiment from a database file.

        Args:
            database: Path to the SQLite database file

        Returns
        -------
            SearchTracker instance with loaded data

        Raises
        ------
            FileNotFoundError: If the database file doesn't exist
        """
        path = Path(database)
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")

        return cls(database=path)

    # ─────────────────────────────────────────────────────────
    # Context Manager
    # ─────────────────────────────────────────────────────────

    def __enter__(self) -> SearchTracker:
        """Return tracker for context manager use."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close tracker when exiting context."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation of the tracker."""
        return (
            f"SearchTracker(name='{self._name}', "
            f"evaluations={self.n_evaluations}, "
            f"runs={len(self.run_ids)})"
        )
