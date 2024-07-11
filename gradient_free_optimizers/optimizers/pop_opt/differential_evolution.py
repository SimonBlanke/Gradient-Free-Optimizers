# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ._individual import Individual


class DifferentialEvolutionOptimizer(BasePopulationOptimizer):
    name = "Differential Evolution"
    _name_ = "differential_evolution"
    __name__ = "DifferentialEvolutionOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(self, *args, population=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.population = population

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        self.offspring_l = []

    def mutation(self, f=1):
        ind_selected = random.sample(self.individuals, 3)

        x_1, x_2, x_3 = [ind.pos_best for ind in ind_selected]
        return x_1 + f * np.subtract(x_2, x_3)

    def crossover(self, target_vector, mutant_vector):
        size = target_vector.size
        vector_l = [target_vector, mutant_vector]

        if random.choice([True, False]):
            choice = [True, False]
        else:
            choice = [False, True]

        add_choice = np.random.randint(2, size=size - 2).astype(bool)
        choice += list(add_choice)
        return np.choose(choice, vector_l)

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        self.p_current = self.individuals[
            self.nth_trial % len(self.individuals)
        ]
        target_vector = self.p_current.pos_new

        mutant_vector = self.mutation()
        pos_new = self.crossover(target_vector, mutant_vector)
        self.p_current.pos_new = self.conv2pos(pos_new)
        return self.p_current.pos_new

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)  # selection
