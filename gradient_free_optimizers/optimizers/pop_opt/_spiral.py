# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..local_opt import HillClimbingOptimizer


def roation(n_dim, vector):
    if n_dim == 1:
        return -1  # not sure about that

    I = np.identity(n_dim - 1)
    R = np.pad(I, ((1, 0), (0, 1)), "minimum")
    R[0, n_dim - 1] = -1

    return np.matmul(R, vector)


class Spiral(HillClimbingOptimizer):
    def __init__(self, *args, decay_rate=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.decay_rate = decay_rate
        self.decay_factor = 1

    def _move_part(self, pos, velo):
        pos_new = (pos + velo).astype(int)
        # limit movement
        n_zeros = [0] * len(self.conv.max_positions)

        return np.clip(pos_new, n_zeros, self.conv.max_positions)

    @HillClimbingOptimizer.track_nth_iter
    @HillClimbingOptimizer.random_restart
    def move_spiral(self, center_pos):
        self.decay_factor *= self.decay_rate
        step_rate = (
            self.decay_factor * (random.random() ** 1 / 3)
            + np.power(self.conv.dim_sizes, 1 / 10) / 2
        )

        print("self.decay_factor", self.decay_factor)

        print("\n  step_rate", step_rate)
        print(
            "  roation",
            roation(len(center_pos), np.subtract(self.pos_current, center_pos)),
        )

        A = center_pos
        B = step_rate * roation(
            len(center_pos), np.subtract(self.pos_current, center_pos)
        )
        print("  B", B)

        new_pos = A + B

        n_zeros = [0] * len(self.conv.max_positions)
        return np.clip(new_pos, n_zeros, self.conv.max_positions).astype(int)

    def evaluate(self, score_new):
        HillClimbingOptimizer.evaluate(self, score_new)
