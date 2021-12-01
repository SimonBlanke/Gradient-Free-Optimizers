# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ._particle import Particle


class ParticleSwarmOptimizer(BasePopulationOptimizer, Search):
    name = "Particle Swarm Optimization"

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

    def iterate(self):
        n_iter = self._iterations(self.particles)
        self.p_current = self.particles[n_iter % len(self.particles)]

        self._sort_best()
        self.p_current.global_pos_best = self.p_sorted[0].pos_best

        pos = self.p_current.iterate()

        return pos

    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)
