from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .optimizers.core_optimizer.converter import Converter


class ResultsManager:
    """Manages optimization results with lazy DataFrame construction.

    This class stores only position indices during optimization, reconstructing
    full parameter dictionaries lazily when the DataFrame is accessed. This
    dramatically reduces memory usage for high-dimensional search spaces.

    """

    def __init__(self, converter: Converter | None = None):
        self._converter = converter
        self._positions: list[tuple[int, ...]] = []
        self._scores: list[float] = []
        self._metrics: list[dict[str, Any]] = []
        self._objectives: list[list[float] | None] = []

    def add(self, result, position) -> None:
        """Add a result with its position (not params dict).

        Parameters
        ----------
        result : Result
            The result object containing score, metrics, and objectives.
        position : array-like
            The position indices (will be converted to tuple for storage).
        """
        self._positions.append(tuple(position))
        self._scores.append(result.score)
        self._metrics.append(result.metrics if result.metrics else {})
        self._objectives.append(result.objectives)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Construct DataFrame lazily from stored positions.

        This reconstructs parameter dictionaries only when needed,
        avoiding the memory cost of storing them during optimization.
        """
        if not self._positions:
            return pd.DataFrame()

        if self._converter is None:
            return pd.DataFrame({"score": self._scores})

        rows = []
        for pos, score, metrics, objectives in zip(
            self._positions, self._scores, self._metrics, self._objectives
        ):
            value = self._converter.position2value(list(pos))
            params = self._converter.value2para(value)

            row: dict[str, Any] = {"score": score, **metrics, **params}
            if objectives is not None:
                for i, obj_val in enumerate(objectives):
                    row[f"objective_{i}"] = obj_val
            rows.append(row)

        return pd.DataFrame(rows)

    def best(self) -> tuple[float, tuple[int, ...]] | None:
        """Return the best score and its position."""
        if not self._scores:
            return None
        best_idx = max(range(len(self._scores)), key=lambda i: self._scores[i])
        return self._scores[best_idx], self._positions[best_idx]
