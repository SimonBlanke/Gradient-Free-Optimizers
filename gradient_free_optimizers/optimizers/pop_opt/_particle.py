# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ..local_opt import HillClimbingOptimizer


class Particle(HillClimbingOptimizer):
    def __init__(
        self,
        *args,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
        rand_rest_p=0.03,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight
        self.rand_rest_p = rand_rest_p

    def _move_part(self, pos, velo):
        pos_new = (pos + velo).astype(int)
        # limit movement
        n_zeros = [0] * len(self.conv.max_positions)

        return np.clip(pos_new, n_zeros, self.conv.max_positions)

    def _move_positioner(self):
        r1, r2 = random.random(), random.random()

        A = self.inertia * self.velo
        B = self.cognitive_weight * r1 * np.subtract(self.pos_best, self.pos_current)
        C = (
            self.social_weight
            * r2
            * np.subtract(self.global_pos_best, self.pos_current)
        )

        new_velocity = A + B + C
        return self._move_part(self.pos_current, new_velocity)

    @HillClimbingOptimizer.track_nth_iter
    @HillClimbingOptimizer.random_restart
    def iterate(self):
        return self._move_positioner()

    def evaluate(self, score_new):
        HillClimbingOptimizer.evaluate(self, score_new)
