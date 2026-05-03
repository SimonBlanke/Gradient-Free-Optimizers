from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..optimizers.core_optimizer.core_optimizer import CoreOptimizer
    from .search_tracker import SearchTracker


class RawData:
    """Direct access to internal tracking lists.

    Exposes raw optimization data with friendly names.
    Score lists are returned by reference (no copy).
    Position lists are converted to parameter dicts on access.

    Accessed internally via ``opt._data.raw`` (private; not part of
    the public API).
    """

    def __init__(self, tracker: SearchTracker, optimizer: CoreOptimizer):
        self._tracker = tracker
        self._optimizer = optimizer

    @property
    def scores_proposed(self) -> list[float]:
        """All scores in order of proposal (one per iteration)."""
        return self._optimizer._score_new_list

    @property
    def scores_accepted(self) -> list[float]:
        """Scores at each accepted (current) position change."""
        return self._optimizer._score_current_list

    @property
    def scores_best(self) -> list[float]:
        """Score at each best-position update."""
        return self._optimizer._score_best_list

    @property
    def positions_proposed(self) -> list[dict]:
        """All proposed positions as parameter dicts.

        Computed on access (converts internal position arrays).
        """
        conv = self._optimizer.conv
        return [
            conv.value2para(conv.position2value(pos))
            for pos in self._optimizer._pos_new_list
        ]

    @property
    def positions_accepted(self) -> list[dict]:
        """Accepted positions as parameter dicts.

        Computed on access (converts internal position arrays).
        """
        conv = self._optimizer.conv
        return [
            conv.value2para(conv.position2value(pos))
            for pos in self._optimizer._pos_current_list
        ]

    @property
    def eval_times(self) -> list[float]:
        """Time spent in objective function per evaluation (seconds)."""
        return self._optimizer.eval_times

    @property
    def iter_times(self) -> list[float]:
        """Total time per iteration including optimizer overhead (seconds)."""
        return self._optimizer.iter_times

    @property
    def convergence(self) -> list[float]:
        """Best score seen at each iteration."""
        return self._tracker.convergence

    @property
    def improvement_iterations(self) -> list[int]:
        """Iteration indices where the best score improved."""
        return self._tracker.improvement_iterations

    @property
    def scores_all(self) -> list[float]:
        """All scores in evaluation order."""
        return self._tracker._scores
