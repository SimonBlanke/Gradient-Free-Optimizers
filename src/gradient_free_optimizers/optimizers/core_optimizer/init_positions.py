# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
from itertools import product

from ..._array_backend import array, power

from .utils import move_random


def _arrays_equal(a, b):
    """Check if two arrays are element-wise equal."""
    if hasattr(a, '__len__') and hasattr(b, '__len__'):
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    return a == b


class Initializer:
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
        return move_random(self.conv.search_space_positions)

    def add_n_random_init_pos(self, n):
        for _ in range(n):
            self.init_positions_l.append(self.move_random())

        self.n_inits = len(self.init_positions_l)

    def get_n_inits(initialize):
        n_inits = 0
        for key_ in initialize.keys():
            init_value = initialize[key_]
            if isinstance(init_value, int):
                n_inits += init_value
            else:
                n_inits += len(init_value)
        return n_inits

    def set_pos(self):
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
                pos = move_random(self.conv.search_space_positions)
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

        n_dim = len(self.conv.max_positions)
        if n_dim > 30:
            positions = []
        else:
            p_per_dim = int(power(n_pos, 1 / n_dim))

            dim_points = []
            for dim in self.conv.max_positions:
                dim_dist = int(dim / (p_per_dim + 1))
                n_points = [n * dim_dist for n in range(1, p_per_dim + 1)]
                dim_points.append(n_points)

            # Use itertools.product instead of meshgrid for n-dimensional grid
            positions = [array(combo) for combo in product(*dim_points)]

        positions_constr = []
        for pos in positions:
            if self.conv.not_in_constraint(pos):
                positions_constr.append(pos)

        return positions_constr

    def _get_random_vertex(self):
        vertex = []
        for dim_positions in self.conv.search_space_positions:
            rnd = random.randint(0, 1)

            if rnd == 0:
                dim_pos = dim_positions[0]
            elif rnd == 1:
                dim_pos = dim_positions[-1]

            vertex.append(dim_pos)
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
                pos = move_random(self.conv.search_space_positions)
                positions.append(pos)

        positions_constr = []
        for pos in positions:
            if self.conv.not_in_constraint(pos):
                positions_constr.append(pos)

        return positions_constr
