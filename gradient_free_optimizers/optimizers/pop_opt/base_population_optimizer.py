# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np

from ..search_tracker import SearchTracker
from ...converter import Converter
from ...results_manager import ResultsManager
from ...optimizers.base_optimizer import get_n_inits
from ...init_positions import Initializer

from ...utils import set_random_seed


class BasePopulationOptimizer:
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__()
        self.conv = Converter(search_space)
        self.results_mang = ResultsManager(self.conv)
        self.initialize = initialize
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process

        self.eval_times = []
        self.iter_times = []

        set_random_seed(nth_process, random_state)

        # get init positions
        init = Initializer(self.conv)
        self.init_positions = init.set_pos(self.initialize)

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter

    def _create_population(self, Optimizer):
        if isinstance(self.population, int):
            population = []
            for pop_ in range(self.population):
                population.append(
                    Optimizer(
                        self.conv.search_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"random": 1},
                    )
                )
        else:
            population = self.population

        n_inits = get_n_inits(self.initialize)

        if n_inits < len(population):
            print("\n Warning: Not enough initial positions for population size")
            print(" Population size is reduced to", n_inits)
            population = population[:n_inits]

        return population

    def finish_initialization(self):
        pass
