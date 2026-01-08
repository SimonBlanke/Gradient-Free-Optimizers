# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from ..._array_backend import (
    array,
    arange,
    abs as np_abs,
    take,
)

# Pandas is still required for DataFrame operations
# TODO: Make pandas optional in Phase 3
import pandas as pd

from functools import reduce
from typing import Any, Callable, TypeVar

from ..._result import Result

# Type alias for array-like types (numpy arrays or GFOArray)
ArrayLike = TypeVar("ArrayLike")


def check_numpy_array(search_space: dict[str, Any]) -> None:
    """Check that search space values are array-like."""
    for para_name, dim_values in search_space.items():
        # Accept numpy arrays or our GFOArray
        has_shape = hasattr(dim_values, "shape")
        has_len = hasattr(dim_values, "__len__")
        if not (has_shape and has_len):
            raise TypeError(
                f"Search space parameter '{para_name}' must be an array-like "
                f"(e.g., numpy.ndarray or list), but got {type(dim_values).__name__}. "
                f"Example: {{'x': np.linspace(-5, 5, 100), 'y': np.arange(0, 10)}}"
            )


class Converter:
    """Converts between position, value, and parameter representations.

    The Converter handles the transformation between three data representations
    used throughout the optimization process:

    - **Position**: Integer indices into the search space arrays (e.g., [2, 5, 1])
    - **Value**: Actual values from the search space (e.g., [0.5, 100, "adam"])
    - **Parameter dict**: Named parameters (e.g., {"lr": 0.5, "batch": 100, "opt": "adam"})

    This class also handles constraint validation and conversion between
    memory dictionaries and DataFrames for warm-starting optimizations.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
        Each array defines the discrete search space for that parameter.
    constraints : list, optional
        List of constraint functions. Each function takes a parameter dict
        and returns True if the constraint is satisfied.

    Attributes
    ----------
    n_dimensions : int
        Number of parameters in the search space.
    search_space : dict
        The original search space dictionary.
    para_names : list
        List of parameter names in order.
    dim_sizes : array
        Number of possible values for each dimension.
    search_space_size : int
        Total number of possible combinations in the search space.
    max_positions : array
        Maximum valid position index for each dimension (dim_sizes - 1).

    Examples
    --------
    >>> search_space = {"x": np.linspace(-5, 5, 100), "y": np.arange(0, 10)}
    >>> conv = Converter(search_space)
    >>> position = [50, 5]
    >>> value = conv.position2value(position)  # [0.0, 5]
    >>> para = conv.value2para(value)  # {"x": 0.0, "y": 5}
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
    ) -> None:
        check_numpy_array(search_space)

        self.n_dimensions = len(search_space)
        self.search_space = search_space

        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

        self.para_names = list(search_space.keys())

        dim_sizes_list = [len(arr) for arr in search_space.values()]
        self.dim_sizes = array(dim_sizes_list)

        # product of list
        self.search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)
        self.max_dim = max(dim_sizes_list)

        self.search_space_positions = [
            list(range(len(arr))) for arr in search_space.values()
        ]
        self.pos_space = dict(
            zip(
                self.para_names,
                [arange(len(arr)) for arr in search_space.values()],
            )
        )

        self.max_positions = self.dim_sizes - 1
        self.search_space_values = list(search_space.values())

    def not_in_constraint(self, position: list[int] | ArrayLike) -> bool:
        """Check if a position satisfies all constraints.

        Parameters
        ----------
        position : list or array
            Position indices to check.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        para = self.value2para(self.position2value(position))

        for constraint in self.constraints:
            if not constraint(para):
                return False
        return True

    def returnNoneIfArgNone(func_: Callable) -> Callable:
        """Decorator that returns None if any argument is None."""

        def wrapper(self, *args: Any) -> Any:
            for arg in [*args]:
                if arg is None:
                    return None
            else:
                return func_(self, *args)

        return wrapper

    @returnNoneIfArgNone
    def position2value(self, position: list[int] | None) -> list[Any] | None:
        """Convert position indices to actual values.

        Parameters
        ----------
        position : list or array
            Integer indices into each dimension of the search space.

        Returns
        -------
        list
            Values from the search space at the given position.
        """
        value = []

        for n, space_dim in enumerate(self.search_space_values):
            value.append(space_dim[position[n]])

        return value

    @returnNoneIfArgNone
    def value2position(self, value: list[Any] | None) -> ArrayLike | None:
        """Convert values to position indices.

        Finds the closest matching position for each value by minimizing
        the absolute difference to search space entries.

        Parameters
        ----------
        value : list
            Values to convert to positions.

        Returns
        -------
        array
            Integer position indices corresponding to the values.
        """
        position = []
        for n, space_dim in enumerate(self.search_space_values):
            # Find index of closest value
            diffs = np_abs(array([value[n] - v for v in space_dim]))
            pos = (
                int(diffs.argmin())
                if hasattr(diffs, "argmin")
                else diffs.tolist().index(min(diffs.tolist()))
            )
            position.append(pos)

        return array(position)

    @returnNoneIfArgNone
    def value2para(self, value: list[Any] | None) -> dict[str, Any] | None:
        """Convert a value list to a parameter dictionary.

        Parameters
        ----------
        value : list
            List of values in parameter order.

        Returns
        -------
        dict
            Dictionary mapping parameter names to values.
        """
        para = {}
        for key, p_ in zip(self.para_names, value):
            para[key] = p_

        return para

    @returnNoneIfArgNone
    def para2value(self, para: dict[str, Any] | None) -> list[Any] | None:
        """Convert a parameter dictionary to a value list.

        Parameters
        ----------
        para : dict
            Dictionary mapping parameter names to values.

        Returns
        -------
        list
            List of values in parameter order.
        """
        value = []
        for para_name in self.para_names:
            value.append(para[para_name])

        return value

    @returnNoneIfArgNone
    def values2positions(self, values: list[list[Any]] | None) -> list[ArrayLike] | None:
        """Convert multiple value lists to position lists.

        Parameters
        ----------
        values : list
            List of value lists to convert.

        Returns
        -------
        list
            List of position arrays.
        """
        positions_temp = []
        values_arr = array(values)

        for n, space_dim in enumerate(self.search_space_values):
            # Get column n from 2D array
            if hasattr(values_arr, "shape") and len(values_arr.shape) > 1:
                values_1d = [values_arr[i, n] for i in range(len(values_arr))]
            else:
                values_1d = [v[n] for v in values]

            # Use searchsorted if available, otherwise manual search
            if hasattr(space_dim, "searchsorted"):
                pos_list = space_dim.searchsorted(values_1d)
            else:
                pos_list = [self._find_position(space_dim, v) for v in values_1d]

            positions_temp.append(pos_list)

        # Transpose and convert to list of arrays
        positions = [
            array(
                [positions_temp[dim][i] for dim in range(len(positions_temp))]
            ).astype(int)
            for i in range(len(positions_temp[0]))
        ]

        return positions

    def _find_position(self, space_dim: Any, value: Any) -> int:
        """Find position using binary search."""
        arr = list(space_dim) if not isinstance(space_dim, list) else space_dim
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @returnNoneIfArgNone
    def positions2values(self, positions: list[ArrayLike] | None) -> list[list[Any]] | None:
        """Convert multiple position lists to value lists.

        Parameters
        ----------
        positions : list
            List of position arrays to convert.

        Returns
        -------
        list
            List of value lists.
        """
        values = []

        for n, space_dim in enumerate(self.search_space_values):
            pos_1d = [pos[n] for pos in positions]
            if hasattr(space_dim, "__getitem__"):
                value_ = [space_dim[p] for p in pos_1d]
            else:
                value_ = take(space_dim, pos_1d)
            values.append(value_)

        values = [list(t) for t in zip(*values)]
        return values

    @returnNoneIfArgNone
    def values2paras(self, values: list[list[Any]]) -> list[dict[str, Any]]:
        """Convert multiple value lists to parameter dictionaries.

        Parameters
        ----------
        values : list
            List of value lists to convert.

        Returns
        -------
        list
            List of parameter dictionaries.
        """
        paras = []
        for value in values:
            paras.append(self.value2para(value))
        return paras

    @returnNoneIfArgNone
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
        value_tuple_list = list(map(tuple, positions))
        # Convert scores to Result objects
        result_objects = [Result(float(score), {}) for score in scores]
        memory_dict = dict(zip(value_tuple_list, result_objects))

        return memory_dict

    @returnNoneIfArgNone
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
        positions = [array(pos).astype(int) for pos in list(memory_dict.keys())]
        # Extract scores from Result objects
        scores = [
            result.score if isinstance(result, Result) else result
            for result in memory_dict.values()
        ]

        return positions, scores

    @returnNoneIfArgNone
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

            print(
                "\nWarning:",
                '"{}"'.format(*missing),
                "is in search_space but not in memory dataframe",
            )
            print("Optimization run will continue without memory warm start\n")

            return {}

    @returnNoneIfArgNone
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
        positions, score = self.memory_dict2positions_scores(memory_dict)
        values = self.positions2values(positions)

        dataframe = pd.DataFrame(values, columns=self.para_names)
        dataframe["score"] = score

        return dataframe
