# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, space_dim, gamma=0.5, **kwargs):
        super().__init__(space_dim, **kwargs)

    def _accept(self):
        f_stun = 1 - np.exp(-self._opt_args_.gamma * self._score_norm())
        return np.exp(-f_stun / self.temp)
