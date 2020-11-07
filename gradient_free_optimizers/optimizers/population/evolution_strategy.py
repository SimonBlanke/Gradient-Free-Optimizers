# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ._individual import Individual


class EvolutionStrategyOptimizer(BasePopulationOptimizer, Search):
    def __init__(
        self,
        search_space,
        mutation_rate=0.7,
        crossover_rate=0.3,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space)

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rand_rest_p = rand_rest_p

        self.individuals = self.optimizers

    def _mutate(self):

        self.p_current = self.individuals[nth_iter % len(self.individuals)]
        pos_new = self.p_current._move_climb(self.p_current.pos_current)

        return pos_new

    def _random_cross(self, array_list):
        n_arrays = len(array_list)
        size = array_list[0].size
        shape = array_list[0].shape

        choice = (
            np.random.randint(n_arrays, size=size).reshape(shape).astype(bool)
        )
        return np.choose(choice, array_list)

    def _sort_best(self):
        scores_list = []
        for ind in self.individuals:
            scores_list.append(ind.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return [self.individuals[idx] for idx in idx_sorted_ind]

    def _cross(self):
        ind_sorted = self._sort_best()

        p_best = ind_sorted[0]
        rnd_int = random.randint(1, len(ind_sorted) - 1)
        p_sec_best = ind_sorted[rnd_int]

        two_best_pos = [p_best.pos_current, p_sec_best.pos_current]
        pos_new = self._random_cross(two_best_pos)

        self.p_current = p_sec_best
        p_sec_best.pos_new = pos_new

        return pos_new

    def _evo_iterate(self):
        total_rate = self.mutation_rate + self.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if len(self.individuals) == 1 or rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            return self._cross()

    def init_pos(self, pos):
        individual = Individual(
            self.conv.search_space, rand_rest_p=self.rand_rest_p
        )
        self.individuals.append(individual)
        individual.init_pos(pos)

        self.p_current = individual

    def iterate(self):
        nth_iter = self._iterations(self.individuals)
        self.p_current = self.individuals[nth_iter % len(self.individuals)]
        return self._evo_iterate()

    def evaluate(self, score_new):
        self.p_current.score_new = score_new

        self.p_current._evaluate_new2current(score_new)
        self.p_current._evaluate_current2best()

