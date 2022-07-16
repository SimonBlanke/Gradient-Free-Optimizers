# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ._particle import Particle


def centeroid(array_list):
    centeroid = []
    for idx in range(array_list[0].shape[0]):
        center_dim_pos = []
        for array in array_list:
            center_dim_pos.append(array[idx])

        center_dim_mean = np.array(center_dim_pos).mean()
        centeroid.append(center_dim_mean)

    return centeroid


class SpiralOptimization(BasePopulationOptimizer, Search):
    name = "Particle Swarm Optimization"
    _name_ = "particle_swarm_optimization"

    def __init__(
        self,
        *args,
        population=10,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.population = population
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight

        self.particles = self._create_population(Particle)
        self.optimizers = self.particles

    def _sort_best(self):
        scores_list = []
        for _p_ in self.particles:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        self.p_sorted = [self.particles[i] for i in idx_sorted_ind]

    def init_pos(self, pos):
        nth_pop = self.nth_iter % len(self.particles)

        self.p_current = self.particles[nth_pop]
        self.p_current.init_pos(pos)

        self.p_current.inertia = self.inertia
        self.p_current.cognitive_weight = self.cognitive_weight
        self.p_current.social_weight = self.social_weight
        self.p_current.temp_weight = self.temp_weight
        self.p_current.rand_rest_p = self.rand_rest_p

        self.p_current.velo = np.zeros(len(self.conv.max_positions))

    def finish_initialization(self):
        self._sort_best()
        self.center_pos = self.p_sorted[0].pos_best
        self.center_score = self.p_sorted[0].score_best

        self.init_done = True

    def iterate(self):
        n_iter = self._iterations(self.particles)
        self.p_current = self.particles[n_iter % len(self.particles)]

        self._sort_best()
        self.p_current.global_pos_best = self.p_sorted[0].pos_best

        return self.p_current.move_spiral(self.center_pos)

    def evaluate(self, score_new):
        if self.init_done and self.p_sorted[0].score_best > self.center_score:
            self.center_pos = self.p_sorted[0].pos_best
            self.center_score = self.p_sorted[0].score_best

        self.p_current.evaluate(score_new)
