# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search


def max_list_idx(list_):
    max_item = max(list_)
    max_item_idx = [i for i, j in enumerate(list_) if j == max_item]
    return max_item_idx[-1:][0]


class PatternSearch(BaseOptimizer, Search):
    name = "Pattern Search"

    def __init__(
        self, *args, n_positions=4, pattern_size=0.25, reduction=0.9, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_positions = n_positions
        self.pattern_size = pattern_size
        self.reduction = reduction

        self.n_positions_ = min(n_positions, self.conv.n_dimensions)
        self.pattern_size_tmp = pattern_size
        self.pattern_pos_l = []

    def generate_pattern(self, current_position):
        pattern_pos_l = []

        n_valid_pos = len(self.positions_valid)
        n_pattern_pos = int(self.n_positions_ * 2)
        n_pos_min = min(n_valid_pos, n_pattern_pos)

        best_in_recent_pos = any(
            np.array_equal(np.array(self.pos_best), pos)
            for pos in self.positions_valid[n_pos_min:]
        )
        if best_in_recent_pos:
            self.pattern_size_tmp *= self.reduction
        pattern_size = self.pattern_size_tmp

        for idx, dim_size in enumerate(self.conv.dim_sizes):
            pos_pattern_p = np.array(current_position)
            pos_pattern_n = np.array(current_position)

            pos_pattern_p[idx] += pattern_size * dim_size
            pos_pattern_n[idx] -= pattern_size * dim_size

            pos_pattern_p = self.conv2pos(pos_pattern_p)
            pos_pattern_n = self.conv2pos(pos_pattern_n)

            pattern_pos_l.append(pos_pattern_p)
            pattern_pos_l.append(pos_pattern_n)

        self.pattern_pos_l = list(random.sample(pattern_pos_l, self.n_positions_))

    @BaseOptimizer.track_nth_iter
    @BaseOptimizer.random_restart
    def iterate(self):
        pos_new = self.pattern_pos_l[0]
        self.pattern_pos_l.pop(0)

        return pos_new

    def finish_initialization(self):
        self.state = "iter"
        self.generate_pattern(self.pos_current)

    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:

            return

        modZero = self.nth_iter % int(self.n_positions_ * 2) == 0

        if modZero or len(self.pattern_pos_l) == 0:
            if self.state == "iter":
                self.generate_pattern(self.pos_current)

            score_new_list_temp = self.scores_valid[-self.n_positions_ :]
            pos_new_list_temp = self.positions_valid[-self.n_positions_ :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)
