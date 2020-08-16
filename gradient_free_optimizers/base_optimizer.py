# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from .search_tracker import SearchTracker


class BaseOptimizer(SearchTracker):
    def __init__(self, space_dim):
        super().__init__()
        self.space_dim = space_dim

        self.init_positions = []

    """
    def _base_init_pos(self, nth_init, positioner):
        init_position = self.init_positions[nth_init]

        self.p_current = positioner
        self.p_current.pos_new = init_position

        # print("init_position", init_position)

        self.p_list.append(self.p_current)

        return init_position

    def _sort_(self):
        self.p_sorted = self.p_list

    def _sort_best(self):
        scores_list = []
        for _p_ in self.p_list:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        self.p_sorted = [self.p_list[i] for i in idx_sorted_ind]

    def _choose_next_pos(self):
        self.p_current = self.p_list[self.nth_iter % len(self.p_list)]
        self.p_rest = [p for p in self.p_list if p is not self.p_current]
    """

    def _base_iterate(self, nth_iter):
        self.nth_iter = nth_iter

    def _evaluate_new2current(self, score_new):
        if score_new >= self.score_current:
            self.score_current = score_new
            self.pos_current = self.pos_new

    def _evaluate_current2best(self):
        if self.score_current >= self.score_best:
            self.score_best = self.score_current
            self.pos_best = self.pos_current

    def _current2best(self):
        self.score_best = self.score_current
        self.pos_best = self.pos_current

    def _new2current(self):
        self.score_current = self.score_new
        self.pos_current = self.pos_new

    def move_random(self):
        pos_new = np.random.uniform(
            np.zeros(self.space_dim.shape), self.space_dim, self.space_dim.shape
        )
        self.pos_new = np.rint(pos_new).astype(int)
        return self.pos_new

    def init_pos(self, init_position):
        self.pos_new = init_position
        return self.pos_new

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()
