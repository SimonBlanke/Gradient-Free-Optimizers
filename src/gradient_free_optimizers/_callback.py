# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CallbackInfo:
    """Immutable snapshot of the optimization state, passed to callbacks.

    Parameters
    ----------
    iteration : int
        Current iteration index (0-based).
    score : float
        Score from the current evaluation.
    params : dict[str, Any]
        Parameters evaluated in the current iteration.
    best_score : float
        Best score found so far.
    best_para : dict[str, Any]
        Parameters corresponding to the best score.
    n_iter : int
        Total number of iterations planned for this search.
    phase : {"init", "iter"}
        Current phase: ``"init"`` during initialization, ``"iter"`` during
        optimization iterations.
    elapsed_time : float
        Seconds elapsed since the search started.
    metrics : dict[str, Any]
        Custom metrics returned by the objective function (empty dict if
        the objective returns only a score).
    convergence : list[float]
        Best score at each iteration so far (read-only copy).
    objectives : list[float] or None
        Raw objective values for multi-objective optimization.
        ``None`` when running single-objective (``n_objectives=1``).
    """

    iteration: int
    score: float
    params: dict[str, Any]
    best_score: float
    best_para: dict[str, Any]
    n_iter: int
    phase: Literal["init", "iter"]
    elapsed_time: float
    metrics: dict[str, Any]
    convergence: list[float]
    objectives: list[float] | None = None
