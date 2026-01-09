# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import copy
import math
import random

from gradient_free_optimizers._array_backend import isinf
from gradient_free_optimizers._array_backend import random as np_random
from gradient_free_optimizers._init_utils import get_default_initialize

from ..local_opt import SimulatedAnnealingOptimizer
from .base_population_optimizer import BasePopulationOptimizer

# Temperature initialization parameters for parallel tempering
# temp = TEMP_BASE ** uniform(0, TEMP_MAX_EXPONENT) gives range [1.0, ~9.3]
# This creates a geometric distribution of temperatures across replicas
TEMP_BASE = 1.1
TEMP_MAX_EXPONENT = 25


class ParallelTemperingOptimizer(BasePopulationOptimizer):
    """Parallel Tempering (Replica Exchange) optimization.

    Runs multiple simulated annealing instances at different temperatures.
    Periodically swaps positions between systems to enable exploration
    at high temperatures and exploitation at low temperatures.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int, default=5
        Number of parallel tempering systems.
    n_iter_swap : int, default=5
        Iterations between temperature swap attempts.
    """

    name = "Parallel Tempering"
    _name_ = "parallel_tempering"
    __name__ = "ParallelTemperingOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=5,
        n_iter_swap=5,
    ):
        if initialize is None:
            initialize = get_default_initialize()

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.population = population
        self.n_iter_swap = n_iter_swap

        self.systems = self._create_population(SimulatedAnnealingOptimizer)
        for system in self.systems:
            system.temp = TEMP_BASE ** random.uniform(0, TEMP_MAX_EXPONENT)
        self.optimizers = self.systems

    def _swap_pos(self):
        for _p1_ in self.systems:
            _systems_temp = copy.copy(self.systems)
            if len(_systems_temp) > 1:
                _systems_temp.remove(_p1_)

            rand = random.uniform(0, 1) * 100
            _p2_ = np_random.choice(_systems_temp)

            p_accept = self._accept_swap(_p1_, _p2_)
            if p_accept > rand:
                _p1_.temp, _p2_.temp = (_p2_.temp, _p1_.temp)

    def _accept_swap(self, _p1_, _p2_):
        denom = _p1_.score_current + _p2_.score_current

        if denom == 0:
            return 100
        elif isinf(abs(denom)):
            return 0
        else:
            score_diff_norm = (_p1_.score_current - _p2_.score_current) / denom

            temp = (1 / _p1_.temp) - (1 / _p2_.temp)
            exponent = score_diff_norm * temp
            try:
                return math.exp(exponent) * 100
            except OverflowError:
                return math.inf

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.systems)
        self.p_current = self.systems[nth_pop]

        return self.p_current.init_pos()

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        """Advance current system and periodically attempt temperature swaps."""
        self.p_current = self.systems[self.nth_trial % len(self.systems)]
        return self.p_current.iterate()

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        notZero = self.n_iter_swap != 0
        modZero = self.nth_trial % self.n_iter_swap == 0

        if notZero and modZero:
            self._swap_pos()

        self.p_current.evaluate(score_new)
