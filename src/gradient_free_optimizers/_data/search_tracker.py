from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..optimizers.core_optimizer.converter import Converter


class SearchTracker:
    """Single collection point for all search tracking data.

    Created by Search at the start of each search() call.
    Fed via track() during the optimization loop.
    Read by SearchData (opt.data) after the search.
    """

    def __init__(self, converter: Converter):
        self._converter = converter

        self._positions: list[tuple[int, ...]] = []
        self._scores: list[float] = []
        self._metrics: list[dict[str, Any]] = []

        self._best_score: float = -math.inf
        self._best_iteration: int = -1
        self.convergence: list[float] = []
        self.improvement_iterations: list[int] = []

        self.n_init: int = 0
        self.n_iter: int = 0

        self.optimizer_name: str = ""
        self.objective_name: str = ""
        self.random_seed: int | None = None

    def track(
        self,
        position,
        score: float,
        metrics: dict[str, Any],
        is_init: bool,
    ) -> None:
        """Record one evaluation. Called once per iteration from Search."""
        self._positions.append(tuple(position))
        self._scores.append(score)
        self._metrics.append(metrics)

        if is_init:
            self.n_init += 1
        else:
            self.n_iter += 1

        if score > self._best_score and not (math.isinf(score) or math.isnan(score)):
            self._best_score = score
            self._best_iteration = self.n_total - 1
            self.improvement_iterations.append(self._best_iteration)

        self.convergence.append(self._best_score)

    @property
    def n_total(self) -> int:
        return self.n_init + self.n_iter

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_iteration(self) -> int:
        return self._best_iteration

    def results_as_dicts(self) -> list[dict[str, Any]]:
        """All results as list of dicts (pandas-free)."""
        rows = []
        for pos, score, metrics in zip(self._positions, self._scores, self._metrics):
            value = self._converter.position2value(list(pos))
            params = self._converter.value2para(value)
            rows.append({"score": score, **metrics, **params})
        return rows
