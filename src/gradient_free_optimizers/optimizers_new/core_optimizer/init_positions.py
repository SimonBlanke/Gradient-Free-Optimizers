# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Initialization strategies for optimizer starting positions."""

import random
from itertools import product

from gradient_free_optimizers._array_backend import array, power
from gradient_free_optimizers._dimension_types import DimensionType

from .utils import move_random


def _arrays_equal(a, b):
    """Check if two arrays are element-wise equal."""
    if hasattr(a, "__len__") and hasattr(b, "__len__"):
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    return a == b


class Initializer:
    """Generate initial positions for optimization algorithms."""

    def __init__(self, conv, initialize):
        self.conv = conv
        self.initialize = initialize

        self.n_inits = 0
        if "random" in initialize:
            self.n_inits += initialize["random"]
        if "grid" in initialize:
            self.n_inits += initialize["grid"]
        if "vertices" in initialize:
            self.n_inits += initialize["vertices"]
        if "warm_start" in initialize:
            self.n_inits += len(initialize["warm_start"])

        self.init_positions_l = None

        self.set_pos()

    def move_random(self):
        """Generate a random position in the search space."""
        return move_random(self.conv.search_space_positions)

    def move_random_typed(self):
        """Generate random position handling all dimension types."""
        if self.conv.is_legacy_mode:
            return self.move_random()

        pos = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            bounds = self.conv.dim_infos[idx].bounds

            if dim_type == DimensionType.CONTINUOUS:
                # Uniform random in continuous range
                pos.append(random.uniform(bounds[0], bounds[1]))
            else:
                # Random index for discrete/categorical
                pos.append(random.randint(int(bounds[0]), int(bounds[1])))

        return array(pos)

    def add_n_random_init_pos(self, n):
        """Add n random initialization positions to the list."""
        for _ in range(n):
            self.init_positions_l.append(self.move_random_typed())

        self.n_inits = len(self.init_positions_l)

    def get_n_inits(initialize):
        """Count total number of initialization positions from config."""
        n_inits = 0
        for key_ in initialize.keys():
            init_value = initialize[key_]
            if isinstance(init_value, int):
                n_inits += init_value
            else:
                n_inits += len(init_value)
        return n_inits

    def set_pos(self):
        """Set initialization positions based on configuration."""
        init_positions_ll = []

        if "random" in self.initialize:
            positions = self._init_random_search(self.initialize["random"])
            init_positions_ll.append(positions)
        if "grid" in self.initialize:
            positions = self._init_grid_search(self.initialize["grid"])
            init_positions_ll.append(positions)
        if "vertices" in self.initialize:
            positions = self._init_vertices(self.initialize["vertices"])
            init_positions_ll.append(positions)
        if "warm_start" in self.initialize:
            positions = self._init_warm_start(self.initialize["warm_start"])
            init_positions_ll.append(positions)

        self.init_positions_l = [
            item for sublist in init_positions_ll for item in sublist
        ]
        self.init_positions_l = self._fill_rest_random(self.init_positions_l)

    def _init_warm_start(self, value_list):
        positions = []

        for value_ in value_list:
            pos = self.conv.value2position(list(value_.values()))
            positions.append(pos)

        positions_constr = []
        for pos in positions:
            if self.conv.not_in_constraint(pos):
                positions_constr.append(pos)

        return positions_constr

    def _init_random_search(self, n_pos):
        positions = []

        if n_pos == 0:
            return positions

        for nth_pos in range(n_pos):
            while True:
                pos = self.move_random_typed()
                if self.conv.not_in_constraint(pos):
                    positions.append(pos)
                    break

        return positions

    def _fill_rest_random(self, positions):
        diff_pos = self.n_inits - len(positions)
        if diff_pos > 0:
            pos_rnd = self._init_random_search(n_pos=diff_pos)

            return positions + pos_rnd
        else:
            return positions

    def _init_grid_search(self, n_pos):
        positions = []

        if n_pos == 0:
            return positions

        n_dim = self.conv.n_dimensions
        if n_dim > 30:
            positions = []
        else:
            p_per_dim = int(power(n_pos, 1 / n_dim))
            if p_per_dim < 1:
                p_per_dim = 1

            dim_points = []
            for idx, dim_type in enumerate(self.conv.dim_types):
                bounds = self.conv.dim_infos[idx].bounds

                if dim_type == DimensionType.CONTINUOUS:
                    # For continuous: evenly spaced points in range
                    min_val, max_val = bounds
                    step = (max_val - min_val) / (p_per_dim + 1)
                    n_points = [min_val + step * n for n in range(1, p_per_dim + 1)]
                else:
                    # For discrete/categorical: evenly spaced indices
                    max_idx = int(bounds[1])
                    if max_idx == 0:
                        n_points = [0]
                    else:
                        dim_dist = int(max_idx / (p_per_dim + 1))
                        if dim_dist < 1:
                            dim_dist = 1
                        n_points = [n * dim_dist for n in range(1, p_per_dim + 1)]
                        # Ensure we don't exceed bounds
                        n_points = [p for p in n_points if p <= max_idx]
                        if not n_points:
                            n_points = [0]

                dim_points.append(n_points)

            # Use itertools.product instead of meshgrid for n-dimensional grid
            positions = [array(combo) for combo in product(*dim_points)]

        positions_constr = []
        for pos in positions:
            if self.conv.not_in_constraint(pos):
                positions_constr.append(pos)

        return positions_constr

    def _get_random_vertex(self):
        """Get a random vertex (corner) of the search space."""
        if self.conv.is_legacy_mode:
            vertex = []
            for dim_positions in self.conv.search_space_positions:
                rnd = random.randint(0, 1)
                if rnd == 0:
                    dim_pos = dim_positions[0]
                else:
                    dim_pos = dim_positions[-1]
                vertex.append(dim_pos)
            return array(vertex)

        # Mixed types: use bounds for vertices
        vertex = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            bounds = self.conv.dim_infos[idx].bounds
            rnd = random.randint(0, 1)

            if dim_type == DimensionType.CONTINUOUS:
                # For continuous, vertex is min or max of range
                vertex.append(bounds[0] if rnd == 0 else bounds[1])
            else:
                # For discrete/categorical, vertex is first or last index
                vertex.append(int(bounds[0]) if rnd == 0 else int(bounds[1]))

        return array(vertex)

    def _init_vertices(self, n_pos):
        positions = []
        for _ in range(n_pos):
            for _ in range(100):
                vertex = self._get_random_vertex()

                # Check if vertex already exists in positions list
                vert_in_list = any(_arrays_equal(vertex, pos) for pos in positions)
                if not vert_in_list:
                    positions.append(vertex)
                    break
            else:
                pos = self.move_random_typed()
                positions.append(pos)

        positions_constr = []
        for pos in positions:
            if self.conv.not_in_constraint(pos):
                positions_constr.append(pos)

        return positions_constr
