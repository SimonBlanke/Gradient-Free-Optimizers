from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    """Internal result container used throughout the search pipeline.

    ``score`` is the scalar fitness used by the optimizer for comparisons.
    ``objectives`` stores the raw objective values when running
    multi-objective optimization (None for single-objective).
    """

    score: float
    metrics: dict
    objectives: list[float] | None = None


@dataclass
class ObjectiveResult:
    """Explicit return type for objective functions.

    Avoids the ambiguity of bare tuples, which becomes critical when the
    return value needs to carry structured data alongside the score (e.g.,
    custom metrics). A plain ``(float, dict)`` tuple works today but
    collides with multi-objective returns like ``(float, float)``.

    Parameters
    ----------
    score : float or list[float]
        Single objective score, or a list of scores for multi-objective.
    metrics : dict, optional
        Custom metrics to record alongside the score.

    Examples
    --------
    Single objective::

        def objective(params):
            loss = params["x"] ** 2
            return ObjectiveResult(score=-loss, metrics={"raw_loss": loss})

    Multi-objective::

        def objective(params):
            x = params["x"]
            return ObjectiveResult(score=[x**2, (x - 2)**2])
    """

    score: float | list[float]
    metrics: dict = field(default_factory=dict)


def unpack_objective_result(raw) -> tuple[float | list[float], dict]:
    """Extract objectives and metrics from an objective function's return.

    Single entry point for parsing objective output. All code paths that
    receive raw objective returns (serial adapter, distributed unpacking,
    minimization wrapper) call this instead of doing their own isinstance
    checks.

    Supported return conventions (checked in this order):

    1. ``ObjectiveResult`` instance (preferred, unambiguous)
    2. ``(score_or_objectives, dict)`` tuple (legacy convention)
    3. ``list`` of floats (multi-objective, no metrics)
    4. ``np.ndarray`` (multi-objective, no metrics)
    5. ``float`` / scalar (single objective, no metrics)

    Parameters
    ----------
    raw : float, list, np.ndarray, tuple, or ObjectiveResult
        Raw return value from an objective function.

    Returns
    -------
    tuple[float | list[float], dict]
        The (objectives, metrics) pair.
    """
    if isinstance(raw, ObjectiveResult):
        return raw.score, raw.metrics
    if isinstance(raw, tuple):
        return raw[0], raw[1]
    if isinstance(raw, list):
        return raw, {}
    if isinstance(raw, np.ndarray) and raw.ndim >= 1 and raw.size > 1:
        return raw.tolist(), {}
    return float(raw), {}


def negate_objectives(objectives: float | list[float]) -> float | list[float]:
    """Negate objectives for minimization (higher-is-better internally)."""
    if isinstance(objectives, list):
        return [-o for o in objectives]
    return -objectives


def objectives_as_list(
    objectives: float | list[float], n_objectives: int
) -> list[float] | None:
    """Convert objectives to a list, or None if single-objective.

    Returns None when n_objectives == 1 to avoid overhead for the
    common single-objective case.
    """
    if n_objectives <= 1:
        return None
    if isinstance(objectives, list):
        return objectives
    return [float(objectives)]
