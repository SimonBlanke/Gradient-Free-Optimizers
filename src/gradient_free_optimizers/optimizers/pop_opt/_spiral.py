# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._array_backend import array, zeros, matmul, clip

from ..local_opt import HillClimbingOptimizer


def rotation(n_dim, vector):
    """Build rotation matrix and apply to vector.

    Creates a rotation matrix R of shape (n_dim, n_dim) where:
    - Identity shifted down by one row (bottom-left block)
    - -1 in top-right corner
    """
    if n_dim == 1:
        return array([-1])  # Return as array for consistency

    # Build rotation matrix manually (equivalent to np.pad on identity)
    # R has shape (n_dim, n_dim)
    # It's the identity matrix (n_dim-1 x n_dim-1) padded with zeros:
    # - 1 row of zeros on top
    # - 1 column of zeros on right
    # - then R[0, n_dim-1] = -1
    R = zeros((n_dim, n_dim))
    for i in range(n_dim - 1):
        R[i + 1, i] = 1.0
    R[0, n_dim - 1] = -1.0

    return matmul(R, array(vector))


class Spiral(HillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.decay_rate = None
        self.decay_factor = 3

    def _move_part(self, pos, velo):
        pos_new = (array(pos) + array(velo)).astype(int)
        # limit movement
        n_zeros = [0] * len(self.conv.max_positions)

        return clip(pos_new, n_zeros, self.conv.max_positions)

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def move_spiral(self, center_pos):
        self.decay_factor *= self.decay_rate
        step_rate = self.decay_factor * array(self.conv.max_positions) / 1000

        A = array(center_pos)
        rot = rotation(len(center_pos), array(self.pos_current) - A)

        B = step_rate * rot
        new_pos = A + B

        n_zeros = [0] * len(self.conv.max_positions)
        pos_new = clip(new_pos, n_zeros, self.conv.max_positions).astype(int)
        return pos_new

    @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new):
        self._new2current()
        self._evaluate_current2best()
