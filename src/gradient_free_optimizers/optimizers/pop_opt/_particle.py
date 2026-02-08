# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random

from gradient_free_optimizers._array_backend import array, clip

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight
        self.rand_rest_p = rand_rest_p

    def _move_part(self, pos, velo):
        pos_new = (array(pos) + array(velo)).astype(int)
        # limit movement
        n_zeros = [0] * len(self.conv.max_positions)

        return clip(pos_new, n_zeros, self.conv.max_positions)

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def move_linear(self):
        r1, r2 = random.random(), random.random()

        A = self.inertia * array(self.velo)
        B = (
            self.cognitive_weight
            * r1
            * (array(self.pos_best) - array(self.pos_current))
        )
        C = (
            self.social_weight
            * r2
            * (array(self.global_pos_best) - array(self.pos_current))
        )

        new_velocity = A + B + C
        return self._move_part(self.pos_current, new_velocity)
