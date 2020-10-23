# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class BasePopulationOptimizer:
    def __init__(self, search_space):
        super().__init__()
        self.search_space = search_space

        self.max_positions = np.array(
            [len(array) - 1 for array in search_space.values()]
        )

        self.eval_times = []
        self.iter_times = []

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter
