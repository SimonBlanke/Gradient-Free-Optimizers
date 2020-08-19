# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer
from ...search import Search


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer, Search):
    def __init__(self, search_space, gamma=0.5, **kwargs):
        super().__init__(search_space, **kwargs)

    def _accept(self):
        f_stun = 1 - np.exp(-self._opt_args_.gamma * self._score_norm())
        return np.exp(-f_stun / self.temp)
