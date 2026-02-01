"""Storage backend protocol definition."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..record import EvaluationRecord, ExperimentMetadata


@runtime_checkable
class StorageBackend(Protocol):
    """
    Protocol that all storage backends must implement.

    This defines the contract for storing and retrieving optimization data.
    Implementations can use SQLite, CSV, JSON, remote databases, etc.
    """

    def connect(self) -> None:
        """
        Establish connection to the storage.

        For file-based backends, this may create the file.
        For database backends, this establishes the connection.
        """
        ...

    def disconnect(self) -> None:
        """
        Close connection to the storage.

        Ensures all data is flushed and resources are released.
        """
        ...

    def append(self, record: EvaluationRecord) -> None:
        """
        Append a single evaluation record.

        Args:
            record: The evaluation record to store
        """
        ...

    def get_records(self, run_id: str | None = None) -> Iterator[EvaluationRecord]:
        """
        Retrieve evaluation records.

        Args:
            run_id: If provided, filter by run ID

        Yields
        ------
            EvaluationRecord objects
        """
        ...

    def save_metadata(self, metadata: ExperimentMetadata) -> None:
        """
        Save experiment metadata.

        Args:
            metadata: The experiment metadata to store
        """
        ...

    def load_metadata(self) -> ExperimentMetadata | None:
        """
        Load experiment metadata.

        Returns
        -------
            The experiment metadata, or None if not found
        """
        ...

    def list_runs(self) -> list[str]:
        """
        List all run IDs in the storage.

        Returns
        -------
            List of run ID strings
        """
        ...

    def count_records(self, run_id: str | None = None) -> int:
        """
        Count the number of records.

        Args:
            run_id: If provided, count only records for this run

        Returns
        -------
            Number of records
        """
        ...
