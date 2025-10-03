# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual


class EvolutionStrategyOptimizer(EvolutionaryAlgorithmOptimizer):
    name = "Evolution Strategy"
    _name_ = "evolution_strategy"
    __name__ = "EvolutionStrategyOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=10,
        offspring=20,
        replace_parents=False,
        mutation_rate=0.7,
        crossover_rate=0.3,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.population = population
        self.offspring = offspring
        self.replace_parents = replace_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

    def _cross(self):
        while True:
            if len(self.individuals) > 2:
                rnd_int2 = random.choice(
                    [
                        i
                        for i in range(0, self.n_ind - 1)
                        if i not in [self.rnd_int]
                    ]
                )
            else:
                rnd_int2 = random.choice(
                    [i for i in range(0, self.n_ind) if i not in [self.rnd_int]]
                )

            p_sec = self.pop_sorted[rnd_int2]
            p_worst = self.pop_sorted[-1]

            two_best_pos = [self.p_current.pos_current, p_sec.pos_current]
            pos_new = self.discrete_recombination(two_best_pos)

            self.p_current = p_worst
            p_worst.pos_new = pos_new

            if self.conv.not_in_constraint(pos_new):
                return pos_new

            return self.p_current.move_climb(pos_new)

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def iterate(self):
        self.n_ind = len(self.individuals)

        if self.n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.sort_pop_best_score()
        self.rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[self.rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            return self._cross()

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)
