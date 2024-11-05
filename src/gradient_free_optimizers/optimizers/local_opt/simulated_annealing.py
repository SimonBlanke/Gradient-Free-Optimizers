# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local_opt import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    name = "Simulated Annealing"
    _name_ = "simulated_annealing"
    __name__ = "SimulatedAnnealingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(self, *args, annealing_rate=0.97, start_temp=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp

    def _p_accept_default(self):
        # the 'minus' is omitted, because we maximize a score
        return np.exp(self._exponent)

    def evaluate(self, score_new):
        StochasticHillClimbingOptimizer.evaluate(self, score_new)
        self.temp *= self.annealing_rate
