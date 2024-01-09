# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer


class OrthogonalGridSearchOptimizer(BaseOptimizer):
    def __init__(self, *args, step_size=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.step_size = step_size

    def grid_move(self):
        mod_tmp = self.nth_trial * self.step_size + int(
            self.nth_trial * self.step_size / self.conv.search_space_size
        )
        div_tmp = self.nth_trial * self.step_size + int(
            self.nth_trial * self.step_size / self.conv.search_space_size
        )
        flipped_new_pos = []

        for dim_size in self.conv.dim_sizes:
            mod = mod_tmp % dim_size
            div = int(div_tmp / dim_size)

            flipped_new_pos.append(mod)

            mod_tmp = div
            div_tmp = div

        return np.array(flipped_new_pos)

    @BaseOptimizer.track_new_pos
    def iterate(self):
        pos_new = self.grid_move()
        pos_new = self.conv2pos(pos_new)
        return pos_new

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
