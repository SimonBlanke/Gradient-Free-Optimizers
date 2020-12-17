# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer
from ...search import Search


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer, Search):
    def __init__(
        self,
        search_space,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        p_accept=0.1,
        norm_factor="adaptive",
        annealing_rate=0.97,
        start_temp=1,
        rand_rest_p=0.03,
    ):
        super().__init__(
            search_space,
            epsilon,
            distribution,
            n_neighbours,
            p_accept,
            norm_factor,
        )
        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp
        self.rand_rest_p = rand_rest_p

    def _accept_default(self):
        return np.exp(-self._score_norm_default() / self.temp)

    def _accept_adapt(self):
        return self._score_norm_adapt() / self.temp

    def evaluate(self, score_new):
        super().evaluate(score_new)

        self.temp = self.temp * self.annealing_rate
