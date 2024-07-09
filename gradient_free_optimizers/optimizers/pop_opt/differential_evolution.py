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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        self.offspring_l = []

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        pass

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)
