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
        epsilon=0.05,
        distribution="normal",
        n_neighbours=5,
        **kwargs
    ):
        super().__init__(search_space, rand_rest_p=0.03)
        self.epsilon = epsilon
        self.distribution = dist_dict[distribution]
        self.n_neighbours = n_neighbours

    def _move_climb(self, pos, epsilon_mod=1):
        sigma = self.space_dim_size * self.epsilon * epsilon_mod
        pos_normal = self.distribution(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(self.space_dim_size)
        pos = np.clip(pos_new_int, n_zeros, self.space_dim_size - 1)

        self.pos_new = pos.astype(int)
        return self.pos_new

    @BaseOptimizer.track_nth_iter
    @BaseOptimizer.random_restart
    def iterate(self):
        return self._move_climb(self.pos_current)

    def evaluate(self, score_new):
        self.score_new = score_new

        modZero = self.nth_iter % self.n_neighbours == 0
        if modZero:
            idx = max_list_idx(self.score_new_list)

            score = self.score_new_list[idx]
            pos = self.pos_new_list[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)

