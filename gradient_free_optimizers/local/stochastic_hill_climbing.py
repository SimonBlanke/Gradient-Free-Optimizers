# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from . import HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.norm_factor = self._opt_args_.norm_factor

        if self.norm_factor == "adaptive":
            self._accept = self._accept_adapt
            self.diff_max = 0
        else:
            self._accept = self._accept_default

    def _consider(self, p_accept):
        rand = random.uniform(0, self._opt_args_.p_down)

        if p_accept > rand:
            self._new2current()

    def _score_norm_default(self, _p_):
        denom = _p_.score_current + _p_.score_new

        if denom == 0:
            return 1
        elif abs(denom) == np.inf:
            return 0
        else:
            return self.norm_factor * (_p_.score_current - _p_.score_new) / denom

    def _score_norm_adapt(self, _p_):
        diff = abs(_p_.score_current - _p_.score_new)
        if self.diff_max < diff:
            self.diff_max = diff

        denom = self.diff_max + diff

        if denom == 0:
            return 1
        elif abs(denom) == np.inf:
            return 0
        else:
            return abs(self.diff_max - diff) / denom

    def _accept_default(self, _p_):
        return np.exp(-self._score_norm_default(_p_))

    def _accept_adapt(self, _p_):
        return self._score_norm_adapt(_p_)

    def _transition(self, score_new):
        if score_new < self.p_current.score_current:
            p_accept = self._accept(self.p_current)
            self._consider(p_accept)

    def evaluate(self, score_new):
        self.p_current.score_new = score_new

        self._evaluate_new2current(score_new)
        self._transition(score_new)
        self._evaluate_current2best()
