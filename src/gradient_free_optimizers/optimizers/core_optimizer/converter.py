# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._array_backend import (
    array,
    arange,
    abs as np_abs,
    take,
)

# Pandas is still required for DataFrame operations
# TODO: Make pandas optional in Phase 3
import pandas as pd

from functools import reduce
from typing import Optional

from gradient_free_optimizers._result import Result


def check_numpy_array(search_space):
    """Check that search space values are array-like."""
    for para_name, dim_values in search_space.items():

        def error_message(wrong_type):
            return "\n Value in '{}' of search space dictionary must be of type array but is '{}' \n".format(
                para_name, wrong_type
            )

        # Accept numpy arrays or our GFOArray
        has_shape = hasattr(dim_values, "shape")
        has_len = hasattr(dim_values, "__len__")
        if not (has_shape and has_len):
            raise ValueError(error_message(type(dim_values)))


class Converter:
    def __init__(self, search_space: dict, constraints: list = None) -> None:
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

    def not_in_constraint(self, position):
        para = self.value2para(self.position2value(position))

        for constraint in self.constraints:
            if not constraint(para):
                return False
        return True

    def returnNoneIfArgNone(func_):
        def wrapper(self, *args):
            for arg in [*args]:
                if arg is None:
                    return None
            else:
                return func_(self, *args)

        return wrapper

    @returnNoneIfArgNone
    def position2value(self, position: Optional[list]) -> Optional[list]:
        value = []

        for n, space_dim in enumerate(self.search_space_values):
            value.append(space_dim[position[n]])

        return value

    @returnNoneIfArgNone
    def value2position(self, value: Optional[list]) -> Optional[list]:
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
    def value2para(self, value: Optional[list]) -> Optional[dict]:
        para = {}
        for key, p_ in zip(self.para_names, value):
            para[key] = p_

        return para

    @returnNoneIfArgNone
    def para2value(self, para: Optional[dict]) -> Optional[list]:
        value = []
        for para_name in self.para_names:
            value.append(para[para_name])

        return value

    @returnNoneIfArgNone
    def values2positions(self, values: Optional[list]) -> Optional[list]:
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

    def _find_position(self, space_dim, value):
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
    def positions2values(self, positions: Optional[list]) -> Optional[list]:
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
    def values2paras(self, values: list) -> list:
        paras = []
        for value in values:
            paras.append(self.value2para(value))
        return paras

    @returnNoneIfArgNone
    def positions_scores2memory_dict(
        self, positions: Optional[list], scores: Optional[list]
    ) -> Optional[dict]:
        value_tuple_list = list(map(tuple, positions))
        # Convert scores to Result objects
        result_objects = [Result(float(score), {}) for score in scores]
        memory_dict = dict(zip(value_tuple_list, result_objects))

        return memory_dict

    @returnNoneIfArgNone
    def memory_dict2positions_scores(self, memory_dict: Optional[dict]):
        positions = [array(pos).astype(int) for pos in list(memory_dict.keys())]
        # Extract scores from Result objects
        scores = [
            result.score if isinstance(result, Result) else result
            for result in memory_dict.values()
        ]

        return positions, scores

    @returnNoneIfArgNone
    def dataframe2memory_dict(
        self, dataframe: Optional[pd.DataFrame]
    ) -> Optional[dict]:
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
        self, memory_dict: Optional[dict]
    ) -> Optional[pd.DataFrame]:
        positions, score = self.memory_dict2positions_scores(memory_dict)
        values = self.positions2values(positions)

        dataframe = pd.DataFrame(values, columns=self.para_names)
        dataframe["score"] = score

        return dataframe
