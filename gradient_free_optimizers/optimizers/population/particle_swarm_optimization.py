# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ._particle import Particle


class ParticleSwarmOptimizer(BasePopulationOptimizer, Search):
    def __init__(
        self,
        search_space,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space)

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight
        self.rand_rest_p = rand_rest_p

        self.particles = self.optimizers

    def _sort_best(self):
        scores_list = []
        for _p_ in self.particles:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        self.p_sorted = [self.particles[i] for i in idx_sorted_ind]

    def init_pos(self, pos):
        particle = Particle(
            self.conv.search_space,
            inertia=self.inertia,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight,
            temp_weight=self.temp_weight,
            rand_rest_p=self.rand_rest_p,
        )
        self.particles.append(particle)
        particle.init_pos(pos)

        self.p_current = particle
        self.p_current.velo = np.zeros(len(self.conv.max_positions))

    def iterate(self):
        n_iter = self._iterations(self.particles)
        self.p_current = self.particles[n_iter % len(self.particles)]

        self._sort_best()
        self.p_current.global_pos_best = self.p_sorted[0].pos_best

        pos = self.p_current.iterate()

        return pos

    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)

