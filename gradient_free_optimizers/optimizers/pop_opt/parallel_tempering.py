# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import copy
import random

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ..local_opt import SimulatedAnnealingOptimizer


class ParallelTemperingOptimizer(BasePopulationOptimizer, Search):
    name = "Parallel Tempering"

    def __init__(self, *args, population=5, n_iter_swap=5, **kwargs):
        super().__init__(*args, **kwargs)

        self.population = population
        self.n_iter_swap = n_iter_swap

        self.systems = self._create_population(SimulatedAnnealingOptimizer)
        for system in self.systems:
            system.temp = 1.1 ** random.uniform(0, 25)
        self.optimizers = self.systems

    def _swap_pos(self):
        for _p1_ in self.systems:
            _systems_temp = copy.copy(self.systems)
            if len(_systems_temp) > 1:
                _systems_temp.remove(_p1_)

            rand = random.uniform(0, 1) * 100
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
            return np.exp(score_diff_norm * temp) * 100

    def init_pos(self, pos):
        nth_pop = self.nth_iter % len(self.systems)

        self.p_current = self.systems[nth_pop]
        self.p_current.init_pos(pos)

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

        self.p_current.evaluate(score_new)
