# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
from scipy.spatial.distance import cdist

import numpy as np
from functools import reduce

from ..smb_opt import SMBO
from ._ant import Ant


class AntColonyOptimization(SMBO):
    name = "Ant Colony Optimization"
    _name_ = "ant_colony_optimization"
    __name__ = "AntColonyOptimization"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(self, p_climb=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_climb = p_climb

        self.trails = {}

        self.ants = self._create_population(Ant)
        self.optimizers = self.ants

    def get_pos_space(self, search_space):
        dim_sizes_list = [len(array) for array in search_space.values()]
        dim_sizes = np.array(dim_sizes_list)

        pos_space = []
        for dim_ in dim_sizes:
            pos_space.append(np.arange(dim_))

        return pos_space

    def _all_possible_pos(self, search_space):
        pos_space = self.get_pos_space(search_space)
        n_dim = len(pos_space)
        return np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

    def create_local_search_space(self, current_position):
        local_ss = {}
        local_range = 10

        for idx, para in enumerate(self.conv.para_names):
            max_dim = max(0, current_position[idx] + local_range)
            min_dim = min(self.conv.dim_sizes[idx], current_position[idx] - local_range)

            dim_pos = np.array(self.conv.search_space_positions[idx])

            dim_pos_center = np.where(
                np.logical_and(dim_pos >= min_dim, dim_pos <= max_dim)
            )[0]
            local_ss[para] = dim_pos_center

        return local_ss

    def get_allowed_movements(self):
        local_ss = self.create_local_search_space(self.ant_current.pos_current)
        return self._all_possible_pos(local_ss)

    def calc_probability(self):
        allowed_movements = self.get_allowed_movements()

        p_denom = 0
        for allowed_movement in allowed_movements:
            movement = (self.ant_current.pos_current, allowed_movement)
            if movement in self.trails:
                trail_level = self.trails[movement][0]
                distance = self.trails[movement][1]

                p_denom += trail_level / distance

    @SMBO.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.ant_current = self.individuals[nth_pop]
        return self.ant_current.init_pos()

    @SMBO.track_new_pos
    def iterate(self):
        self.ant_current = self.ants[self.nth_trial % len(self.ants)]

        if random.uniform(0, 1) < self.p_climb:
            return self.ant_current.iterate()

    @SMBO.track_new_score
    def evaluate(self, score_new):
        improvement = self.ant_current.score_current - self.ant_current.score_new
        distance = cdist(self.ant_current.pos_current, self.ant_current.pos_new)
        movement = (self.ant_current.pos_current, self.ant_current.pos_new)
        self.trails[movement] = (improvement, distance)

        self.ant_current.evaluate(score_new)
