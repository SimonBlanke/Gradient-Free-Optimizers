# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer
from ...search import Search


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer, Search):
    def __init__(self, search_space, annealing_rate=0.975, start_temp=1):
        super().__init__(search_space)
        self.annealing_rate = annealing_rate
        self.temp = start_temp

    def _accept_default(self):
        return np.exp(-self._score_norm_default() / self.temp)

    def _accept_adapt(self):
        return self._score_norm_adapt() / self.temp

    def evaluate(self, score_new):
        super().evaluate(score_new)

        self.temp = self.temp * self.annealing_rate
