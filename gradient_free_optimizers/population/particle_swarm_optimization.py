# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ..local import HillClimbingOptimizer
from ..base_positioner import BasePositioner


class ParticleSwarmOptimizer(HillClimbingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.n_positioners = self._opt_args_.n_particles

    def _move_positioner(self):
        r1, r2 = random.random(), random.random()

        A = self._opt_args_.inertia * self.p_current.velo
        B = (
            self._opt_args_.cognitive_weight
            * r1
            * np.subtract(self.p_current.pos_best, self.p_current.pos_new)
        )
        C = (
            self._opt_args_.social_weight
            * r2
            * np.subtract(self.global_pos_best, self.p_current.pos_new)
        )

        new_velocity = A + B + C

        self.p_current.velo = new_velocity
        return self.p_current.move_part(self.p_current.pos_current)

    def init_pos(self, nth_init):
        pos_new = self._base_init_pos(
            nth_init, Particle(self.space_dim, self._opt_args_)
        )

        self.p_current.velo = np.zeros(len(self.space_dim))

        return pos_new

    def iterate(self, nth_iter):
        self._base_iterate(nth_iter)
        self._sort_best()
        self._choose_next_pos()
        self.global_pos_best = self.p_sorted[0].pos_best
        pos = self._move_positioner()

        return pos


class Particle(BasePositioner):
    def __init__(self, space_dim, _opt_args_):
        super().__init__(space_dim, _opt_args_)
        self.velo = None

    def move_part(self, pos):
        pos_new = (pos + self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(self.space_dim)
        self.pos_new = np.clip(pos_new, n_zeros, self.space_dim)

        return self.pos_new
