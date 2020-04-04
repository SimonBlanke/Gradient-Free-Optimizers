# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ..base_optimizer import BaseOptimizer
from ..base_positioner import BasePositioner


class ParticleSwarmOptimizer(BaseOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_positioners = self._opt_args_.n_particles

    def _init_particle(self, _cand_):
        _p_ = Particle()
        _p_.pos_new = _cand_._space_.get_random_pos()
        _p_.velo = np.zeros(len(_cand_._space_.search_space))

        self._optimizer_eval(_cand_, _p_)
        self._update_pos(_cand_, _p_)

        return _p_

    def _move_positioner(self, _cand_, _p_):
        r1, r2 = random.random(), random.random()

        A = self._opt_args_.inertia * _p_.velo
        B = (
            self._opt_args_.cognitive_weight
            * r1
            * np.subtract(_p_.pos_best, _p_.pos_new)
        )
        C = (
            self._opt_args_.social_weight
            * r2
            * np.subtract(_cand_.pos_best, _p_.pos_new)
        )

        new_velocity = A + B + C

        _p_.velo = new_velocity
        _p_.move_part(_cand_, _p_.pos_new)

    def _iterate(self, i, _cand_):
        _p_current = self.p_list[i % self.n_positioners]
        self._move_positioner(_cand_, _p_current)

        self._optimizer_eval(_cand_, _p_current)
        self._update_pos(_cand_, _p_current)

        return _cand_

    def _init_iteration(self, _cand_):
        p = self._init_particle(_cand_)

        self._optimizer_eval(_cand_, p)
        self._update_pos(_cand_, p)

        return p


class Particle(BasePositioner):
    def __init__(self):
        super().__init__(self)
        self.velo = None

    def move_part(self, _cand_, pos):
        pos_new = (pos + self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(_cand_._space_.dim)
        self.pos_new = np.clip(pos_new, n_zeros, _cand_._space_.dim)
