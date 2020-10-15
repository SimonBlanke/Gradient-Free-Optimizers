# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


class Initializer:
    def __init__(self, conv):
        self.conv = conv

    def set_pos(self, initialize):
        init_positions_list = []

        if "random" in initialize:
            positions = self._init_random_search(initialize["random"])
            init_positions_list.append(positions)
        if "grid" in initialize:
            positions = self._init_grid_search(initialize["grid"])
            init_positions_list.append(positions)
        if "vertices" in initialize:
            positions = self._init_vertices(initialize["vertices"])
            init_positions_list.append(positions)
        if "warm_start" in initialize:
            positions = self._init_warm_start(initialize["warm_start"])
            init_positions_list.append(positions)

        return [item for sublist in init_positions_list for item in sublist]

    def _init_warm_start(self, value_list):
        positions = []

        for value_ in value_list:
            pos = self.conv.value2position(list(value_.values()))
            positions.append(pos)

        return positions

    def move_random(self):
        position = []
        for search_space_pos in self.conv.search_space_positions:
            pos_ = random.choice(search_space_pos)
            position.append(pos_)

        return np.array(position)

    def _init_random_search(self, n_pos):
        positions = []

        if n_pos == 0:
            return positions

        for nth_pos in range(n_pos):
            pos = self.move_random()
            positions.append(pos)

        return positions

    def _fill_rest_random(self, n_pos, positions):
        diff_pos = n_pos - len(positions)
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
        p_per_dim = int(np.power(n_pos, 1 / n_dim))

        for dim in self.conv.max_positions:
            dim_dist = int(dim / (p_per_dim + 1))
            n_points = [n * dim_dist for n in range(1, p_per_dim + 1)]

            positions.append(n_points)

        pos_mesh = np.array(np.meshgrid(*positions))
        positions = list(pos_mesh.T.reshape(-1, n_dim))

        positions = self._fill_rest_random(n_pos, positions)

        return positions

    def _init_vertices(self, n_pos):
        positions = []

        if n_pos == 0:
            return positions

        # zero_array = np.zeros(self.conv.max_positions.shape)
        sub_arrays = []

        for dim in self.conv.max_positions:
            sub_array = np.array([0, dim])
            sub_arrays.append(sub_array)

        n_dims = len(self.conv.max_positions)
        pos_comb_np = list(
            np.array(np.meshgrid(*sub_arrays)).T.reshape(-1, n_dims)
        )
        k = min(len(pos_comb_np), n_pos)

        positions = random.sample(pos_comb_np, k)

        positions = self._fill_rest_random(n_pos, positions)

        return positions

