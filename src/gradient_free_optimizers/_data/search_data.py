from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any

from .raw_data import RawData

if TYPE_CHECKING:
    from ..optimizers.core_optimizer.core_optimizer import CoreOptimizer
    from .search_tracker import SearchTracker


class SearchData:
    """Read-only view on search results with computed metrics.

    Provides derived metrics from SearchTracker (search-level data)
    and CoreOptimizer history lists (optimizer-level data).

    Accessed via ``opt.data`` property after calling ``search()``.
    """

    def __init__(self, tracker: SearchTracker, optimizer: CoreOptimizer):
        self._tracker = tracker
        self._optimizer = optimizer
        self._raw: RawData | None = None

    @property
    def n_iter(self) -> int:
        """Total number of iterations executed (init + optimization)."""
        return self._tracker.n_total

    @property
    def n_init(self) -> int:
        """Number of initialization iterations."""
        return self._tracker.n_init

    @property
    def n_optimization(self) -> int:
        """Number of optimization iterations (after init phase)."""
        return self._tracker.n_iter

    @property
    def best_score(self) -> float:
        """Best score found during the search."""
        return self._tracker.best_score

    @property
    def best_para(self) -> dict:
        """Parameters corresponding to the best score."""
        return self._optimizer.best_para

    @property
    def best_iteration(self) -> int:
        """Iteration index at which the best score was found."""
        return self._tracker.best_iteration

    @property
    def total_time(self) -> float:
        """Total wall time in seconds."""
        return sum(self._optimizer.iter_times)

    @property
    def eval_time(self) -> float:
        """Total time spent in the objective function (seconds)."""
        return sum(self._optimizer.eval_times)

    @property
    def overhead_time(self) -> float:
        """Time spent in optimizer logic (seconds)."""
        return self.total_time - self.eval_time

    @property
    def overhead_pct(self) -> float:
        """Optimizer overhead as percentage of total time."""
        if self.total_time == 0:
            return 0.0
        return (self.overhead_time / self.total_time) * 100

    @property
    def eval_pct(self) -> float:
        """Evaluation time as percentage of total time."""
        if self.total_time == 0:
            return 0.0
        return (self.eval_time / self.total_time) * 100

    @property
    def avg_eval_time(self) -> float:
        """Average time per objective function evaluation (seconds)."""
        times = self._optimizer.eval_times
        if not times:
            return 0.0
        return self.eval_time / len(times)

    @property
    def avg_iter_time(self) -> float:
        """Average time per iteration including overhead (seconds)."""
        times = self._optimizer.iter_times
        if not times:
            return 0.0
        return self.total_time / len(times)

    @property
    def throughput(self) -> float:
        """Iterations per second (0 if total_time is 0)."""
        if self.total_time == 0:
            return 0.0
        return self.n_iter / self.total_time

    @property
    def n_score_improvements(self) -> int:
        """Number of times the best score improved."""
        return len(self._tracker.improvement_iterations)

    @property
    def longest_plateau(self) -> tuple[int, int, int]:
        """Longest stretch without improvement.

        Returns
        -------
        tuple of (length, start_iter, end_iter)
        """
        impr = self._tracker.improvement_iterations
        n = self._tracker.n_total

        if n == 0:
            return (0, 0, 0)

        if not impr:
            return (n, 0, n - 1)

        boundaries = [-1] + impr + [n - 1]
        longest = 0
        start = 0
        end = 0
        for i in range(len(boundaries) - 1):
            gap = boundaries[i + 1] - boundaries[i]
            if gap > longest:
                longest = gap
                start = boundaries[i] + 1
                end = boundaries[i + 1]
        return (longest, start, end)

    @property
    def n_invalid(self) -> int:
        """Number of evaluations that returned inf or nan."""
        return sum(1 for s in self._tracker._scores if math.isinf(s) or math.isnan(s))

    @property
    def score_min(self) -> float:
        """Minimum score across all evaluations (excludes inf/nan)."""
        valid = [
            s for s in self._tracker._scores if not (math.isinf(s) or math.isnan(s))
        ]
        return min(valid) if valid else float("nan")

    @property
    def score_max(self) -> float:
        """Maximum score across all evaluations (excludes inf/nan)."""
        valid = [
            s for s in self._tracker._scores if not (math.isinf(s) or math.isnan(s))
        ]
        return max(valid) if valid else float("nan")

    @property
    def score_mean(self) -> float:
        """Mean score across all evaluations (excludes inf/nan)."""
        valid = [
            s for s in self._tracker._scores if not (math.isinf(s) or math.isnan(s))
        ]
        return statistics.mean(valid) if valid else float("nan")

    @property
    def score_std(self) -> float:
        """Standard deviation of scores (excludes inf/nan)."""
        valid = [
            s for s in self._tracker._scores if not (math.isinf(s) or math.isnan(s))
        ]
        if len(valid) < 2:
            return 0.0
        return statistics.stdev(valid)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed positions that were accepted (0.0 to 1.0)."""
        n_proposed = len(self._optimizer._pos_new_list)
        n_accepted = len(self._optimizer._pos_current_list)
        if n_proposed == 0:
            return 0.0
        return n_accepted / n_proposed

    @property
    def convergence_data(self) -> list[float]:
        """Best score at each iteration (for plotting convergence curves)."""
        return self._tracker.convergence

    @property
    def results(self) -> list[dict[str, Any]]:
        """All evaluated positions as list of dicts (pandas-free).

        Each dict contains ``score``, any custom metrics, and all
        parameter values.
        """
        return self._tracker.results_as_dicts()

    @property
    def raw(self) -> RawData:
        """Direct access to internal tracking lists."""
        if self._raw is None:
            self._raw = RawData(self._tracker, self._optimizer)
        return self._raw
