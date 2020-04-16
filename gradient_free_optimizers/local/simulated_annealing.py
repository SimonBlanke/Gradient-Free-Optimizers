# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.temp = self._opt_args_.start_temp

    # use _consider from StochasticHillClimbingOptimizer

    def _accept_default(self, _p_):
        return np.exp(-self._score_norm_default(_p_) / self.temp)

    def _accept_adapt(self, _p_):
        return self._score_norm_adapt(_p_) * self.temp

    def _iterate(self, i, _cand_):
        self._stochastic_hill_climb_iter(i, _cand_)

        self.temp = self.temp * self._opt_args_.annealing_rate

        return _cand_
