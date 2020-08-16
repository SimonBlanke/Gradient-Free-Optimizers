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


class HillClimbingOptimizer(BaseOptimizer):
    def __init__(
        self, space_dim, epsilon=0.05, distribution="normal", n_neighbours=1,
    ):
        super().__init__(space_dim)
        self.epsilon = epsilon
        self.distribution = dist_dict[distribution]
        self.n_neighbours = n_neighbours

    def _move_climb(self, pos, epsilon_mod=1):
        sigma = self.space_dim * self.epsilon * epsilon_mod
        pos_normal = self.distribution(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(self.space_dim)
        pos = np.clip(pos_new_int, n_zeros, self.space_dim)

        self.pos_new = pos.astype(int)
        return self.pos_new

    def iterate(self):
        return self._move_climb(self.pos_current)

