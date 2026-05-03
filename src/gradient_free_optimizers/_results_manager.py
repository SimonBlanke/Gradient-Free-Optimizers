from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from .optimizers.core_optimizer.converter import Converter


class ResultsManager:
    """Manages optimization results with lazy DataFrame construction.

    This class stores only position indices during optimization, reconstructing
    full parameter dictionaries lazily when the DataFrame is accessed. This
    dramatically reduces memory usage for high-dimensional search spaces.

    When conditions are active, inactive parameters appear as NaN in the
    DataFrame, making it easy to filter rows by parameter availability.
    """

    def __init__(self, converter: Converter | None = None):
        self._converter = converter
        self._positions: list[tuple[int, ...]] = []
        self._scores: list[float] = []
        self._metrics: list[dict[str, Any]] = []
        self._active_masks: list[dict[str, bool] | None] = []

    def add(self, result, position, active_mask=None) -> None:
        """Add a result with its position (not params dict).

        Parameters
        ----------
        result : Result
            The result object containing score and metrics.
        position : array-like
            The position indices (will be converted to tuple for storage).
        active_mask : dict[str, bool] or None
            Which parameters were active for this evaluation. None means
            all parameters were active (no conditions).
        """
        self._positions.append(tuple(position))
        self._scores.append(result.score)
        self._metrics.append(result.metrics if result.metrics else {})
        self._active_masks.append(active_mask)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Construct DataFrame lazily from stored positions.

        This reconstructs parameter dictionaries only when needed,
        avoiding the memory cost of storing them during optimization.
        Inactive parameters (from conditions) appear as NaN.
        """
        import pandas as pd

        if not self._positions:
            return pd.DataFrame()

        if self._converter is None:
            return pd.DataFrame({"score": self._scores})

        rows = []
        for pos, score, metrics, active_mask in zip(
            self._positions, self._scores, self._metrics, self._active_masks
        ):
            value = self._converter.position2value(list(pos))
            params = self._converter.value2para(value)

            if active_mask is not None:
                for param_name, is_active in active_mask.items():
                    if not is_active:
                        params[param_name] = math.nan

            row = {"score": score, **metrics, **params}
            rows.append(row)

        return pd.DataFrame(rows)

    def best(self) -> tuple[float, tuple[int, ...]] | None:
        """Return the best score and its position."""
        if not self._scores:
            return None
        best_idx = max(range(len(self._scores)), key=lambda i: self._scores[i])
        return self._scores[best_idx], self._positions[best_idx]
