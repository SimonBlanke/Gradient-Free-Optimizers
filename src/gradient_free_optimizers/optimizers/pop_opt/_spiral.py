# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..local_opt import HillClimbingOptimizer


def roation(n_dim, vector):
    if n_dim == 1:
        return -1  # not sure about that

    I = np.identity(n_dim - 1)
    R = np.pad(I, ((1, 0), (0, 1)), constant_values=(0, 0))
    R[0, n_dim - 1] = -1

    return np.matmul(R, vector)


class Spiral(HillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.decay_rate = None
        self.decay_factor = 3

    def _move_part(self, pos, velo):
        pos_new = (pos + velo).astype(int)
        # limit movement
        n_zeros = [0] * len(self.conv.max_positions)

        return np.clip(pos_new, n_zeros, self.conv.max_positions)

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def move_spiral(self, center_pos):
        self.decay_factor *= self.decay_rate
        step_rate = self.decay_factor * self.conv.max_positions / 1000

        A = center_pos
        rot = roation(len(center_pos), np.subtract(self.pos_current, center_pos))

        B = np.multiply(step_rate, rot)
        new_pos = A + B

        n_zeros = [0] * len(self.conv.max_positions)
        pos_new = np.clip(new_pos, n_zeros, self.conv.max_positions).astype(int)
        return pos_new

    @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new):
        self._new2current()
        self._evaluate_current2best()
