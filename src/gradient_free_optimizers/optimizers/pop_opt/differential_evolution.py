# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual


class DifferentialEvolutionOptimizer(EvolutionaryAlgorithmOptimizer):
    name = "Differential Evolution"
    _name_ = "differential_evolution"
    __name__ = "DifferentialEvolutionOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        *args,
        population=10,
        mutation_rate=0.9,
        crossover_rate=0.9,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.population = population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        self.offspring_l = []

    def mutation(self, f=1):
        ind_selected = random.sample(self.individuals, 3)

        x_1, x_2, x_3 = [ind.pos_best for ind in ind_selected]
        return x_1 + self.mutation_rate * np.subtract(x_2, x_3)

    def _constraint_loop(self, position):
        while True:
            if self.conv.not_in_constraint(position):
                return position
            position = self.p_current.move_climb(position, epsilon_mod=0.3)

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def iterate(self):
        self.p_current = self.individuals[
            self.nth_trial % len(self.individuals)
        ]
        target_vector = self.p_current.pos_new

        mutant_vector = self.mutation()

        crossover_rates = [1 - self.crossover_rate, self.crossover_rate]
        pos_new = self.discrete_recombination(
            [target_vector, mutant_vector],
            crossover_rates,
        )
        pos_new = self.conv2pos(pos_new)
        pos_new = self._constraint_loop(pos_new)

        self.p_current.pos_new = self.conv2pos(pos_new)
        return self.p_current.pos_new

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)  # selection
