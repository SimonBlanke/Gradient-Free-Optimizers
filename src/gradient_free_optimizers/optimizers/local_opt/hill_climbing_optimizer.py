# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer

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


class HillClimbingOptimizer(BaseOptimizer):
    name = "Hill Climbing"
    _name_ = "hill_climbing"
    __name__ = "HillClimbingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        *args,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours

    def move_climb(self, pos, epsilon_mod=1):
        while True:
            sigma = self.conv.max_positions * self.epsilon * epsilon_mod
            pos_normal = dist_dict[self.distribution](pos, sigma, pos.shape)
            pos = self.conv2pos(pos_normal)

            if self.conv.not_in_constraint(pos):
                return pos
            epsilon_mod *= 1.01

    @BaseOptimizer.track_new_pos
    @BaseOptimizer.random_iteration
    def iterate(self):
        return self.move_climb(self.pos_current)

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:
            return

        modZero = self.nth_trial % self.n_neighbours == 0
        if modZero:
            score_new_list_temp = self.scores_valid[-self.n_neighbours :]
            pos_new_list_temp = self.positions_valid[-self.n_neighbours :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)
