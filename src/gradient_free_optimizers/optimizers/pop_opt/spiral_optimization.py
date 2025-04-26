# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ._spiral import Spiral


def centeroid(array_list):
    centeroid = []
    for idx in range(array_list[0].shape[0]):
        center_dim_pos = []
        for array in array_list:
            center_dim_pos.append(array[idx])

        center_dim_mean = np.array(center_dim_pos).mean()
        centeroid.append(center_dim_mean)

    return centeroid


class SpiralOptimization(BasePopulationOptimizer):
    name = "Spiral Optimization"
    _name_ = "spiral_optimization"
    __name__ = "SpiralOptimization"

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
        decay_rate=0.99,
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
        self.decay_rate = decay_rate

        self.particles = self._create_population(Spiral)
        self.optimizers = self.particles

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.particles)

        self.p_current = self.particles[nth_pop]
        self.p_current.decay_rate = self.decay_rate

        return self.p_current.init_pos()

    def finish_initialization(self):
        self.sort_pop_best_score()
        self.center_pos = self.pop_sorted[0].pos_current
        self.center_score = self.pop_sorted[0].score_current

        self.search_state = "iter"

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        while True:
            self.p_current = self.particles[
                self.nth_trial % len(self.particles)
            ]

            self.sort_pop_best_score()
            self.p_current.global_pos_best = self.pop_sorted[0].pos_current

            pos_new = self.p_current.move_spiral(self.center_pos)

            if self.conv.not_in_constraint(pos_new):
                return pos_new
            return self.p_current.iterate()

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        if self.search_state == "iter":
            if self.pop_sorted[0].score_current > self.center_score:
                self.center_pos = self.pop_sorted[0].pos_current
                self.center_score = self.pop_sorted[0].score_current

        self.p_current.evaluate(score_new)
