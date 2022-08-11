# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from ..local_opt import HillClimbingOptimizer


def roation(n_dim, vector):
    if n_dim == 1:
        return -1  # not sure about that

    I = np.identity(n_dim - 1)
    print("  I", I)

    R = np.pad(I, ((1, 0), (0, 1)), constant_values=(0, 0))
    R[0, n_dim - 1] = -1

    print("  R", R)

    return np.matmul(R, vector)


class Spiral(HillClimbingOptimizer):
    def __init__(self, *args, decay_rate=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.decay_rate = decay_rate
        self.decay_factor = 3

    def _move_part(self, pos, velo):
        pos_new = (pos + velo).astype(int)
        # limit movement
        n_zeros = [0] * len(self.conv.max_positions)

        return np.clip(pos_new, n_zeros, self.conv.max_positions)

    @HillClimbingOptimizer.track_nth_iter
    @HillClimbingOptimizer.random_restart
    def move_spiral(self, center_pos):
        self.decay_factor *= self.decay_rate
        # TODO step rate in N dimensions!
        step_rate = self.decay_factor * self.conv.max_positions / 1000
        """
        step_rate = (
            self.decay_factor * (random.random() ** 1 / 3)
            + np.power(self.conv.dim_sizes, 1 / 10) / 5
        )
        """

        A = center_pos
        rot = roation(len(center_pos), np.subtract(self.pos_current, center_pos))

        dist_ = cdist(self.pos_current.reshape(1, -1), center_pos.reshape(1, -1))

        """
        if dist_ < 1:
            pos = self.move_random()
            self.pos_current = pos
            return pos
        """
        # rot = np.maximum()

        B = np.multiply(step_rate, rot)
        print("\n  A", A)
        print("  B", B)
        print("  rot", rot)
        print("  self.pos_current", self.pos_current)
        print("  center_pos", center_pos)

        print(
            "  np.subtract(self.pos_current, center_pos)",
            np.subtract(self.pos_current, center_pos),
        )

        print("  step_rate", step_rate)

        new_pos = A + B

        n_zeros = [0] * len(self.conv.max_positions)
        pos_new = np.clip(new_pos, n_zeros, self.conv.max_positions).astype(int)
        print("  pos_new", pos_new)

        return pos_new

    def evaluate(self, score_new):
        self.score_new = score_new

        self._new2current()
        self._evaluate_current2best()
