# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Any, TypeVar

from gradient_free_optimizers._array_backend import (
    abs as np_abs,
)
from gradient_free_optimizers._array_backend import (
    arange,
    array,
    take,
)
from gradient_free_optimizers._dimension_types import (
    DimensionInfo,
    DimensionMasks,
    DimensionType,
    classify_search_space_value,
)

from ._converter_memory import MemoryOperationsMixin

# Type alias for array-like types (numpy arrays or GFOArray)
ArrayLike = TypeVar("ArrayLike")


def check_search_space_value(search_space: dict[str, Any]) -> None:
    """Check that search space values are valid.

    Valid types:
    - Array-like (numpy.ndarray, GFOArray) for discrete numerical dimensions
    - Python list for categorical dimensions
    - Tuple of (min, max) for continuous dimensions
    """
    for para_name, dim_values in search_space.items():
        dim_type = classify_search_space_value(dim_values)

        if dim_type == DimensionType.CONTINUOUS:
            # Tuple (min, max) - validate it has exactly 2 numeric elements
            if not isinstance(dim_values, tuple) or len(dim_values) != 2:
                raise TypeError(
                    f"Continuous dimension '{para_name}' must be a tuple of "
                    f"(min, max), but got {type(dim_values).__name__}."
                )
            min_val, max_val = dim_values
            if not (
                isinstance(min_val, (int | float))
                and isinstance(max_val, (int | float))
            ):
                raise TypeError(
                    f"Continuous dimension '{para_name}' bounds must be numeric, "
                    f"but got ({type(min_val).__name__}, {type(max_val).__name__})."
                )
            if min_val >= max_val:
                raise ValueError(
                    f"Continuous dimension '{para_name}' min ({min_val}) must be "
                    f"less than max ({max_val})."
                )

        elif dim_type == DimensionType.CATEGORICAL:
            # Python list - validate it has at least one element
            if not isinstance(dim_values, list):
                raise TypeError(
                    f"Categorical dimension '{para_name}' must be a list, "
                    f"but got {type(dim_values).__name__}."
                )
            if len(dim_values) == 0:
                raise ValueError(
                    f"Categorical dimension '{para_name}' must have at least "
                    f"one value."
                )

        else:  # DISCRETE_NUMERICAL
            # Accept numpy arrays or our GFOArray
            has_shape = hasattr(dim_values, "shape")
            has_len = hasattr(dim_values, "__len__")
            if not (has_shape and has_len):
                raise TypeError(
                    f"Search space parameter '{para_name}' must be an array-like "
                    f"(e.g., numpy.ndarray), list, or tuple (min, max), "
                    f"but got {type(dim_values).__name__}. "
                    f"Examples: np.linspace(-5, 5, 100), ['a', 'b', 'c'], (0.0, 1.0)"
                )
            if len(dim_values) == 0:
                raise ValueError(
                    f"Discrete dimension '{para_name}' must have at least one value."
                )


# Backward compatibility alias
def check_numpy_array(search_space: dict[str, Any]) -> None:
    """Check that search space values are array-like.

    .. deprecated::
        Use check_search_space_value instead, which supports all dimension types.
    """
    check_search_space_value(search_space)


class Converter(MemoryOperationsMixin):
    """Converts between position, value, and parameter representations.

    The Converter handles the transformation between three data representations
    used throughout the optimization process:

    - **Position**: Integer indices into the search space arrays (e.g., [2, 5, 1])
    - **Value**: Actual values from the search space (e.g., [0.5, 100, "adam"])
    - **Parameter dict**: Named parameters (e.g., {"lr": 0.5, "batch": 100})

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
        check_search_space_value(search_space)

        self.n_dimensions = len(search_space)
        self.search_space = search_space

        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

        self.para_names = list(search_space.keys())

        # Analyze dimension types (new functionality)
        self._analyze_dimension_types()

        # For backward compatibility: compute dim_sizes for discrete dimensions
        # For continuous dimensions, we use a placeholder size
        dim_sizes_list = self._compute_dim_sizes_list()
        self.dim_sizes = array(dim_sizes_list)

        # product of list (only meaningful for fully discrete spaces)
        self.search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)
        self.max_dim = max(dim_sizes_list)

        # search_space_positions: for discrete/categorical, list of valid indices
        # for continuous, we use an empty list (not applicable)
        self.search_space_positions = self._compute_search_space_positions()

        self.pos_space = dict(
            zip(
                self.para_names,
                [arange(size) if size > 0 else arange(1) for size in dim_sizes_list],
            )
        )

        self.max_positions = self.dim_sizes - 1
        self.search_space_values = self._compute_search_space_values()

    def _analyze_dimension_types(self) -> None:
        """Analyze and classify all dimensions by type.

        Populates dim_types, dim_infos, and dim_masks attributes.
        """
        self.dim_types: list[DimensionType] = []
        self.dim_infos: list[DimensionInfo] = []

        for name, value in self.search_space.items():
            dim_type = classify_search_space_value(value)
            self.dim_types.append(dim_type)

            if dim_type == DimensionType.CONTINUOUS:
                min_val, max_val = value
                info = DimensionInfo(
                    name=name,
                    dim_type=dim_type,
                    bounds=(float(min_val), float(max_val)),
                    values=None,
                    size=None,
                )
            elif dim_type == DimensionType.CATEGORICAL:
                info = DimensionInfo(
                    name=name,
                    dim_type=dim_type,
                    bounds=(0, len(value) - 1),
                    values=list(value),
                    size=len(value),
                )
            else:  # DISCRETE_NUMERICAL
                values = list(value)
                info = DimensionInfo(
                    name=name,
                    dim_type=dim_type,
                    bounds=(0, len(values) - 1),
                    values=values,
                    size=len(values),
                )

            self.dim_infos.append(info)

        self.dim_masks = DimensionMasks.from_dim_types(self.dim_types)

    def _compute_dim_sizes_list(self) -> list[int]:
        """Compute dimension sizes for backward compatibility.

        For continuous dimensions, returns a placeholder value of 1.
        """
        sizes = []
        for info in self.dim_infos:
            if info.dim_type == DimensionType.CONTINUOUS:
                # Placeholder for continuous - actual range is in bounds
                sizes.append(1)
            else:
                sizes.append(info.size)
        return sizes

    def _compute_search_space_positions(self) -> list[list[int]]:
        """Compute valid position indices for each dimension."""
        positions = []
        for info in self.dim_infos:
            if info.dim_type == DimensionType.CONTINUOUS:
                # Continuous dimensions don't have discrete positions
                # Use empty list as placeholder
                positions.append([])
            else:
                positions.append(list(range(info.size)))
        return positions

    def _compute_search_space_values(self) -> list:
        """Compute search space values list for backward compatibility.

        For continuous dimensions, returns the (min, max) tuple.
        For discrete/categorical, returns the values list.
        """
        values = []
        for idx, info in enumerate(self.dim_infos):
            if info.dim_type == DimensionType.CONTINUOUS:
                # Store the tuple for continuous
                values.append(self.search_space[info.name])
            else:
                # Store the array/list for discrete/categorical
                values.append(self.search_space[info.name])
        return values

    @property
    def is_legacy_mode(self) -> bool:
        """Check if search space contains only discrete-numerical dimensions.

        When True, the optimizer can use original code paths for
        full backward compatibility.
        """
        return self.dim_masks.is_homogeneous_discrete

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
            For continuous dimensions, the position is the actual value.

        Returns
        -------
        list
            Values from the search space at the given position.
        """
        value = []

        for n, space_dim in enumerate(self.search_space_values):
            if self.dim_types[n] == DimensionType.CONTINUOUS:
                # For continuous dimensions, position is the actual value
                value.append(float(position[n]))
            else:
                # For discrete/categorical, position is an index
                value.append(space_dim[int(position[n])])

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
            Position array. For discrete/categorical, integer indices.
            For continuous, the actual float values.
        """
        position = []
        for n, space_dim in enumerate(self.search_space_values):
            if self.dim_types[n] == DimensionType.CONTINUOUS:
                # For continuous dimensions, the value is the position
                position.append(float(value[n]))
            else:
                # Find index of closest value for discrete/categorical
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
    def values2positions(
        self, values: list[list[Any]] | None
    ) -> list[ArrayLike] | None:
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
    def positions2values(
        self, positions: list[ArrayLike] | None
    ) -> list[list[Any]] | None:
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
