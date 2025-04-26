# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ..base_optimizer import BaseOptimizer
from ..local_opt.hill_climbing_optimizer import HillClimbingOptimizer


def max_list_idx(list_):
    max_item = max(list_)
    max_item_idx = [i for i, j in enumerate(list_) if j == max_item]
    return max_item_idx[-1:][0]


class PatternSearch(BaseOptimizer):
    name = "Pattern Search"
    _name_ = "pattern_search"
    __name__ = "PatternSearch"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        n_positions=4,
        pattern_size=0.25,
        reduction=0.9,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

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

        self.pattern_pos_l = list(
            random.sample(pattern_pos_l, self.n_positions_)
        )

    @BaseOptimizer.track_new_pos
    @BaseOptimizer.random_iteration
    def iterate(self):
        while True:
            pos_new = self.pattern_pos_l[0]
            self.pattern_pos_l.pop(0)

            if self.conv.not_in_constraint(pos_new):
                return pos_new
            return self.move_climb(pos_new)

    def finish_initialization(self):
        self.generate_pattern(self.pos_current)
        self.search_state = "iter"

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:
            return

        modZero = self.nth_trial % int(self.n_positions_ * 2) == 0

        if modZero or len(self.pattern_pos_l) == 0:
            if self.search_state == "iter":
                self.generate_pattern(self.pos_current)

            score_new_list_temp = self.scores_valid[-self.n_positions_ :]
            pos_new_list_temp = self.positions_valid[-self.n_positions_ :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)
