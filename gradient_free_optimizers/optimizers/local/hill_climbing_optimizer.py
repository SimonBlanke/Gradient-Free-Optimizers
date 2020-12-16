# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search

from numpy.random import normal, laplace, logistic, gumbel

dist_dict = {
    "normal": normal,
    "laplace": laplace,
    "logistic": logistic,
    "gumbel": gumbel,
}


def max_list_idx(list_):
    max_item = max(list_)
    max_item_idx = [i for i, j in enumerate(list_) if j == max_item]
    return max_item_idx[-1:][0]


class HillClimbingOptimizer(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        rand_rest_p=0.01,
    ):
        super().__init__(search_space)
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.rand_rest_p = rand_rest_p

    def _move_climb(self, pos, epsilon_mod=1):
        sigma = self.conv.max_positions * self.epsilon * epsilon_mod
        pos_normal = dist_dict[self.distribution](pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(self.conv.max_positions)
        pos_new = np.clip(pos_new_int, n_zeros, self.conv.max_positions)

        return pos_new.astype(int)

    @BaseOptimizer.track_nth_iter
    @BaseOptimizer.random_restart
    def iterate(self):
        return self._move_climb(self.pos_current)

    def evaluate(self, score_new):
        self.score_new = score_new

        modZero = self.nth_iter % self.n_neighbours == 0
        if modZero:
            score_new_list_temp = self.score_new_list[-self.n_neighbours :]
            pos_new_list_temp = self.pos_new_list[-self.n_neighbours :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)

