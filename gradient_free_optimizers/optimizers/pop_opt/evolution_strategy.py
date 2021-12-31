# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ._individual import Individual


class EvolutionStrategyOptimizer(BasePopulationOptimizer, Search):
    name = "Evolution Strategy"

    def __init__(
        self, *args, population=10, mutation_rate=0.7, crossover_rate=0.3, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.population = population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

    def _random_cross(self, array_list):
        n_arrays = len(array_list)
        size = array_list[0].size

        choice = [True, False]
        if size > 2:
            add_choice = np.random.randint(n_arrays, size=size - 2).astype(bool)
            choice += list(add_choice)

        cross_array = np.choose(choice, array_list)
        return cross_array

    def _sort_best(self):
        scores_list = []
        for ind in self.individuals:
            scores_list.append(ind.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return [self.individuals[idx] for idx in idx_sorted_ind]

    def _cross(self):
        if len(self.individuals) > 2:
            rnd_int2 = random.choice(
                [i for i in range(0, self.n_ind - 1) if i not in [self.rnd_int]]
            )
        else:
            rnd_int2 = random.choice(
                [i for i in range(0, self.n_ind) if i not in [self.rnd_int]]
            )

        p_sec = self.ind_sorted[rnd_int2]
        p_worst = self.ind_sorted[-1]

        two_best_pos = [self.p_current.pos_current, p_sec.pos_current]
        pos_new = self._random_cross(two_best_pos)

        self.p_current = p_worst
        p_worst.pos_new = pos_new

        return pos_new

    def init_pos(self, pos):
        nth_pop = self.nth_iter % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        self.p_current.init_pos(pos)

    def iterate(self):
        self.n_ind = len(self.individuals)

        if self.n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.ind_sorted = self._sort_best()
        self.rnd_int = random.randint(0, len(self.ind_sorted) - 1)
        self.p_current = self.ind_sorted[self.rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            return self._cross()

    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)
