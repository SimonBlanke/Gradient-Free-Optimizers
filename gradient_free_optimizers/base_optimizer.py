# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from .search_tracker import SearchTracker


class BaseOptimizer(SearchTracker):
    def __init__(self, space_dim):
        super().__init__()
        self.space_dim = space_dim

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

    def init_pos(self, pos):
        self.pos_new = pos
        return self.pos_new

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()
