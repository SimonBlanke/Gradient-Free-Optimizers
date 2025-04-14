# gradient_free_optimizers/hilbert_grid_search.py
# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from numpy_hilbert_curve import decode
from ..base_optimizer import BaseOptimizer


class HilbertGridSearchOptimizer(BaseOptimizer):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        step_size=1,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.step_size = step_size
        self.Z = 0  # Current Hilbert integer
        self.valid_count = 0  # Counter for valid points

    def hilbert_move(self):
        while True:
            # Decode the current Hilbert integer to get an nD point
            point = decode(np.array([self.Z]), self.conv.n_dim, self.conv.n_dim)[0]
            self.Z += 1
            # Check if the point is within the grid bounds
            if all(point[i] < self.conv.dim_sizes[i] for i in range(self.conv.n_dim)):
                self.valid_count += 1
                # Take every step_size-th valid point
                if self.valid_count % self.step_size == 1:
                    return np.array(point)
            # Continue if point is out of bounds

    @BaseOptimizer.track_new_pos
    def iterate(self):
        pos_new = self.hilbert_move()
        pos_new = self.conv2pos(pos_new)
        return pos_new

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)