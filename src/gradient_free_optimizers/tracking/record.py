"""Data models for the tracking module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EvaluationRecord:
    """
    A single evaluation record from an optimization run.

    Attributes
    ----------
        iteration: The iteration number (0-indexed)
        parameters: Dictionary of parameter names to values
        score: The objective function score
        timestamp: When the evaluation completed
        evaluation_time: How long the evaluation took (seconds)
        run_id: Identifier for the run (for multi-run experiments)
    """

    iteration: int
    parameters: dict[str, Any]
    score: float
    timestamp: datetime
    evaluation_time: float
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "parameters": self.parameters,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "evaluation_time": self.evaluation_time,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRecord:
        """Create from dictionary."""
        return cls(
            iteration=data["iteration"],
            parameters=data["parameters"],
            score=data["score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            evaluation_time=data["evaluation_time"],
            run_id=data["run_id"],
        )


@dataclass
class ExperimentMetadata:
    """
    Metadata for an experiment.

    Attributes
    ----------
        name: Experiment name
        created_at: When the experiment was created
        search_space: Optional search space definition
        description: Optional description
        tags: Optional key-value tags
        schema_version: For future compatibility
    """

    name: str
    created_at: datetime = field(default_factory=datetime.now)
    search_space: dict[str, Any] | None = None
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "search_space": self.search_space,
            "description": self.description,
            "tags": self.tags,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetadata:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            search_space=data.get("search_space"),
            description=data.get("description"),
            tags=data.get("tags", {}),
            schema_version=data.get("schema_version", 1),
        )
