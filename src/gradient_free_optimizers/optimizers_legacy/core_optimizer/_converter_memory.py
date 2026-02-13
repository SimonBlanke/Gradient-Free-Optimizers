# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Memory operations mixin for Converter class.

This module isolates pandas-dependent operations for converting between
memory dictionaries and DataFrames. This separation enables future work
to make pandas an optional dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gradient_free_optimizers._array_backend import array
from gradient_free_optimizers._result import Result

if TYPE_CHECKING:
    import pandas as pd

    from .converter import ArrayLike

logger = logging.getLogger(__name__)


class MemoryOperationsMixin:
    """Mixin providing memory/DataFrame conversion methods for Converter.

    This mixin adds methods for converting between:
    - Position/score lists and memory dictionaries
    - Memory dictionaries and pandas DataFrames

    These operations are used for warm-starting optimizations from
    previous results and for exporting optimization history.

    Note: This mixin requires the host class to have:
    - `search_space`: dict mapping parameter names to value arrays
    - `para_names`: list of parameter names
    - `values2positions()`: method to convert values to positions
    - `positions2values()`: method to convert positions to values
    """

    # These will be provided by the Converter class
    search_space: dict[str, Any]
    para_names: list[str]

    def positions_scores2memory_dict(
        self, positions: list[ArrayLike] | None, scores: list[float] | None
    ) -> dict[tuple[int, ...], Result] | None:
        """Convert positions and scores to a memory dictionary.

        Parameters
        ----------
        positions : list
            List of position arrays.
        scores : list
            List of corresponding scores.

        Returns
        -------
        dict
            Dictionary mapping position tuples to Result objects.
        """
        if positions is None or scores is None:
            return None

        value_tuple_list = list(map(tuple, positions))
        # Convert scores to Result objects
        result_objects = [Result(float(score), {}) for score in scores]
        memory_dict = dict(zip(value_tuple_list, result_objects))

        return memory_dict

    def memory_dict2positions_scores(
        self, memory_dict: dict[tuple[int, ...], Result] | None
    ) -> tuple[list[ArrayLike], list[float]] | None:
        """Convert a memory dictionary to positions and scores.

        Parameters
        ----------
        memory_dict : dict
            Dictionary mapping position tuples to Result objects.

        Returns
        -------
        tuple
            (positions, scores) where positions is a list of arrays
            and scores is a list of floats.
        """
        if memory_dict is None:
            return None

        positions = [array(pos).astype(int) for pos in list(memory_dict.keys())]
        # Extract scores from Result objects
        scores = [
            result.score if isinstance(result, Result) else result
            for result in memory_dict.values()
        ]

        return positions, scores

    def dataframe2memory_dict(
        self, dataframe: pd.DataFrame | None
    ) -> dict[tuple[int, ...], Result] | None:
        """Convert a pandas DataFrame to a memory dictionary.

        Used for warm-starting from previous optimization results.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame with columns for each parameter and a 'score' column.

        Returns
        -------
        dict
            Memory dictionary, or empty dict if parameters don't match.
        """
        if dataframe is None:
            return None

        parameter = set(self.search_space.keys())
        memory_para = set(dataframe.columns)

        if parameter <= memory_para:
            values = list(dataframe[self.para_names].values)
            positions = self.values2positions(values)
            scores = dataframe["score"]

            memory_dict = self.positions_scores2memory_dict(positions, scores)

            return memory_dict
        else:
            missing = parameter - memory_para

            logger.warning(
                '"%s" is in search_space but not in memory dataframe. '
                "Optimization run will continue without memory warm start.",
                next(iter(missing)),
            )

            return {}

    def memory_dict2dataframe(
        self, memory_dict: dict[tuple[int, ...], Result] | None
    ) -> pd.DataFrame | None:
        """Convert a memory dictionary to a pandas DataFrame.

        Parameters
        ----------
        memory_dict : dict
            Memory dictionary mapping position tuples to Result objects.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each parameter and a 'score' column.
        """
        if memory_dict is None:
            return None

        # Import pandas here to keep it optional at module level
        import pandas as pd

        positions, score = self.memory_dict2positions_scores(memory_dict)
        values = self.positions2values(positions)

        dataframe = pd.DataFrame(values, columns=self.para_names)
        dataframe["score"] = score

        return dataframe
