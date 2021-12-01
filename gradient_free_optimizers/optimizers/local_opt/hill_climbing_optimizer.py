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
    name = "Hill Climbing"

    def __init__(
        self, *args, epsilon=0.03, distribution="normal", n_neighbours=3, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours

    def _move_climb(self, pos, epsilon_mod=1):
        sigma = self.conv.max_positions * self.epsilon * epsilon_mod
        pos_normal = dist_dict[self.distribution](pos, sigma, pos.shape)

        return self.conv2pos(pos_normal)

    @BaseOptimizer.track_nth_iter
    @BaseOptimizer.random_restart
    def iterate(self):
        return self._move_climb(self.pos_current)

    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:
            return

        modZero = self.nth_iter % self.n_neighbours == 0
        if modZero:
            score_new_list_temp = self.scores_valid[-self.n_neighbours :]
            pos_new_list_temp = self.positions_valid[-self.n_neighbours :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)
