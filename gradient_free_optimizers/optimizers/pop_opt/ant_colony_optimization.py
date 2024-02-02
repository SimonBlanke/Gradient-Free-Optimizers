# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
from scipy.spatial.distance import cdist

from .base_population_optimizer import BasePopulationOptimizer
from ._ant import Ant


class AntColonyOptimization(BasePopulationOptimizer):
    name = "Ant Colony Optimization"
    _name_ = "ant_colony_optimization"
    __name__ = "AntColonyOptimization"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(self, p_climb=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_climb = p_climb

        self.trails = {}

        self.ants = self._create_population(Ant)
        self.optimizers = self.ants

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.ant_current = self.individuals[nth_pop]
        return self.ant_current.init_pos()

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        self.ant_current = self.ants[self.nth_trial % len(self.ants)]

        if random.uniform(0, 1) < self.p_climb:
            return self.ant_current.iterate()

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        improvement = self.ant_current.score_current - self.ant_current.score_new
        distance = cdist(self.ant_current.pos_current, self.ant_current.pos_new)
        movement = (self.ant_current.pos_current, self.ant_current.pos_new)
        self.trails[movement] = (improvement, distance)

        self.ant_current.evaluate(score_new)
