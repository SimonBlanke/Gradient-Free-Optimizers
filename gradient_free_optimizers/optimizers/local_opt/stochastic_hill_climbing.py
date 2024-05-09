# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from random import random

from . import HillClimbingOptimizer
from ..core_optimizer.parameter_tracker.stochastic_hill_climbing import ParameterTracker


class StochasticHillClimbingOptimizer(HillClimbingOptimizer, ParameterTracker):
    name = "Stochastic Hill Climbing"
    _name_ = "stochastic_hill_climbing"
    __name__ = "StochasticHillClimbingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(self, *args, p_accept=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.p_accept = p_accept
        self.temp = 1

    @ParameterTracker.considered_transitions
    def _consider(self, p_accept):
        if p_accept >= random():
            self._execute_transition()

    @ParameterTracker.transitions
    def _execute_transition(self):
        self._new2current()

    @property
    def _normalized_energy_state(self):
        denom = self.score_current + self.score_new

        if denom == 0:
            return 1
        elif abs(denom) == np.inf:
            return 0
        else:
            return (self.score_new - self.score_current) / denom

    @property
    def _exponent(self):
        if self.temp == 0:
            return -np.inf
        else:
            return self._normalized_energy_state / self.temp

    def _p_accept_default(self):
        return self.p_accept * 2 / (1 + np.exp(self._exponent))

    def _transition(self, score_new):
        if score_new <= self.score_current:
            p_accept = self._p_accept_default()
            self._consider(p_accept)

    def evaluate(self, score_new):
        self._transition(score_new)
        HillClimbingOptimizer.evaluate(self, score_new)
