# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)

    def _accept(self, _p_):
        f_stun = 1 - np.exp(-self._opt_args_.gamma * self._score_norm(_p_))
        return np.exp(-f_stun / self.temp)
