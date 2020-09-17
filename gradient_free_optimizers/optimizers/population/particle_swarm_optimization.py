# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ...search import Search
from ..local import HillClimbingOptimizer


class ParticleSwarmOptimizer(BasePopulationOptimizer, Search):
    def __init__(
        self,
        search_space,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
    ):
        super().__init__(search_space)

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight

        self.particles = self.optimizers

    def _move_part(self, pos, velo):
        pos_new = (pos + velo).astype(int)
        # limit movement
        n_zeros = [0] * len(self.space_dim_size)
        self.p_current.pos_new = np.clip(pos_new, n_zeros, self.space_dim_size)

        return self.p_current.pos_new

    def _move_positioner(self):
        r1, r2 = random.random(), random.random()

        A = self.inertia * self.p_current.velo
        B = (
            self.cognitive_weight
            * r1
            * np.subtract(self.p_current.pos_best, self.p_current.pos_current)
        )
        C = (
            self.social_weight
            * r2
            * np.subtract(self.global_pos_best, self.p_current.pos_current)
        )

        new_velocity = A + B + C
        return self._move_part(self.p_current.pos_current, new_velocity)

    def _sort_best(self):
        scores_list = []
        for _p_ in self.particles:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        self.p_sorted = [self.particles[i] for i in idx_sorted_ind]

    def init_pos(self, pos):
        particle = HillClimbingOptimizer(self.search_space)
        self.particles.append(particle)
        particle.init_pos(pos)

        self.p_current = particle
        self.p_current.velo = np.zeros(len(self.space_dim_size))

    def iterate(self):
        n_iter = self._iterations(self.particles)
        self.p_current = self.particles[n_iter % len(self.particles)]

        self._sort_best()
        self.global_pos_best = self.p_sorted[0].pos_best

        if self.temp_weight > random.uniform(0, 1):
            pos = self.p_current._move_climb(self.p_current.pos_current)
        else:
            pos = self._move_positioner()

        return pos

    def evaluate(self, score_new):
        self.p_current.score_new = score_new

        self.p_current._evaluate_new2current(score_new)
        self.p_current._evaluate_current2best()

