# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    def __init__(self, space_dim, annealing_rate=0.99, start_temp=100):
        super().__init__(space_dim)
        self.annealing_rate = annealing_rate
        self.temp = start_temp

    # use _consider from StochasticHillClimbingOptimizer

    def _accept_default(self):
        return np.exp(-self._score_norm_default() / self.temp)

    def _accept_adapt(self):
        return self._score_norm_adapt() * self.temp

    def evaluate(self, score_new):
        super().evaluate(score_new)

        self.temp = self.temp * self.annealing_rate
