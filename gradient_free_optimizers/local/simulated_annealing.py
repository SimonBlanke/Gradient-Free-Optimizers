# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.temp = self._opt_args_.start_temp

    # use _consider from StochasticHillClimbingOptimizer

    def _accept_default(self, _p_):
        return np.exp(-self._score_norm_default(_p_) / self.temp)

    def _accept_adapt(self, _p_):
        return self._score_norm_adapt(_p_) * self.temp

    def evaluate(self, score_new):
        super().evaluate(score_new)

        self.temp = self.temp * self._opt_args_.annealing_rate
