# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ..local import SimulatedAnnealingOptimizer


class ParallelTemperingOptimizer(BasePopulationOptimizer, Search):
    def __init__(self, search_space, n_iter_swap=10, rand_rest_p=0.03):
        super().__init__(search_space)

        self.n_iter_swap = n_iter_swap
        self.rand_rest_p = rand_rest_p

        self.systems = self.optimizers

    def _swap_pos(self):
        _systems_temp = self.systems[:]

        for _p1_ in self.systems:
            rand = random.uniform(0, 1)
            _p2_ = np.random.choice(_systems_temp)

            p_accept = self._accept_swap(_p1_, _p2_)
            if p_accept > rand:
                _p1_.temp, _p2_.temp = (_p2_.temp, _p1_.temp)

    def _accept_swap(self, _p1_, _p2_):
        denom = _p1_.score_current + _p2_.score_current

        if denom == 0:
            return 100
        elif abs(denom) == np.inf:
            return 0
        else:
            score_diff_norm = (_p1_.score_current - _p2_.score_current) / denom

            temp = (1 / _p1_.temp) - (1 / _p2_.temp)
            return np.exp(score_diff_norm * temp)

    def init_pos(self, pos):
        system = SimulatedAnnealingOptimizer(
            self.conv.search_space, rand_rest_p=self.rand_rest_p
        )
        self.systems.append(system)
        system.init_pos(pos)

        self.p_current = system

    def iterate(self):
        nth_iter = self._iterations(self.systems)
        self.p_current = self.systems[nth_iter % len(self.systems)]

        return self.p_current.iterate()

    def evaluate(self, score_new):
        nth_iter = self._iterations(self.systems)

        notZero = self.n_iter_swap != 0
        modZero = nth_iter % self.n_iter_swap == 0

        if notZero and modZero:
            self._swap_pos()

        self.p_current.score_new = score_new

        self.p_current._evaluate_new2current(score_new)
        self.p_current._evaluate_current2best()

