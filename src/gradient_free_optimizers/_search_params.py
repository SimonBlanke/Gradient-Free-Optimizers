"""SearchParams: dict subclass passed to objective functions.

Provides read-only optimization context (iteration, best score, etc.)
and methods to dynamically update conditions and constraints during a run.
Backward-compatible with plain dict usage.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class SearchParams(dict):
    """Parameter dictionary with optimization context and runtime control.

    Behaves identically to a regular ``dict`` for all standard operations
    (indexing, iteration, ``len``, ``in``, etc.). Additionally exposes
    read-only properties for the current optimization state and methods
    to update conditions/constraints that take effect on the next iteration.

    In distributed mode, the context properties return snapshotted values
    from the time the position was dispatched to the worker.

    Parameters
    ----------
    data : dict
        The active parameter key/value pairs for this iteration.
    optimizer_ref : object or None
        Reference to the optimizer instance (provides live context and
        the Converter for deferred updates). None in distributed mode.
    context_snapshot : dict or None
        Pre-computed context values for distributed workers where a live
        optimizer reference is not available (cross-process boundary).
    """

    _optimizer = None
    _n_iter = None
    _iteration = None
    _search_space = None
    _phase = None
    _best_score = None
    _best_params = None

    def __init__(
        self,
        data: dict[str, Any],
        optimizer_ref: Any = None,
        context_snapshot: dict[str, Any] | None = None,
    ):
        super().__init__(data)
        self._optimizer_ref = optimizer_ref
        self._snapshot = context_snapshot
        self._deferred: list[tuple[str, Any]] = []

    @property
    def iteration(self) -> int:
        """Current iteration index (0-based)."""
        if self._optimizer_ref is not None:
            return getattr(self._optimizer_ref, "nth_iter", 0)
        if self._snapshot is not None:
            return self._snapshot.get("iteration", 0)
        return 0

    @property
    def score_best(self) -> float:
        """Best score found so far."""
        if self._optimizer_ref is not None:
            return getattr(self._optimizer_ref, "_score_best", -math.inf)
        if self._snapshot is not None:
            return self._snapshot.get("score_best", -math.inf)
        return -math.inf

    @property
    def pos_best(self) -> dict[str, Any] | None:
        """Best parameters found so far, as a dict.

        Returns None if no evaluation has been performed yet. In
        distributed mode, returns the snapshotted best parameters
        from dispatch time.
        """
        if self._optimizer_ref is not None:
            opt = self._optimizer_ref
            pos = getattr(opt, "_pos_best", None)
            if pos is None:
                return None
            value = opt.conv.position2value(pos)
            return opt.conv.value2para(value)
        if self._snapshot is not None:
            return self._snapshot.get("pos_best")
        return None

    @property
    def n_iter_total(self) -> int:
        """Total number of iterations completed so far."""
        if self._optimizer_ref is not None:
            return getattr(self._optimizer_ref, "n_iter_total", 0)
        if self._snapshot is not None:
            return self._snapshot.get("n_iter_total", 0)
        return 0

    def set_conditions(self, conditions: list[Callable]) -> None:
        """Add conditions that take effect from the next iteration.

        Each condition function receives a full parameter dict and returns
        a ``dict[str, bool]`` mapping parameter names to active/inactive.
        These are appended to the existing conditions list (additive).

        In distributed mode (process-based), updates apply only within
        the current worker process.

        Parameters
        ----------
        conditions : list[callable]
            Condition functions to add.
        """
        self._deferred.append(("conditions", conditions))

    def set_constraints(self, constraints: list[Callable]) -> None:
        """Add constraints that take effect from the next iteration.

        Each constraint function receives a parameter dict and returns
        ``True`` if the constraint is satisfied. These are appended to
        the existing constraints list (additive).

        In distributed mode (process-based), updates apply only within
        the current worker process.

        Parameters
        ----------
        constraints : list[callable]
            Constraint functions to add.
        """
        self._deferred.append(("constraints", constraints))

    def _apply_deferred(self) -> None:
        """Apply all deferred updates to the optimizer.

        Called by the ObjectiveAdapter after the objective function returns.
        Modifications are additive: new conditions/constraints are appended.
        """
        if not self._deferred or self._optimizer_ref is None:
            return

        for kind, value in self._deferred:
            if kind == "conditions":
                self._optimizer_ref.conv.conditions.extend(value)
            elif kind == "constraints":
                self._optimizer_ref.conv.constraints.extend(value)

        self._deferred.clear()
