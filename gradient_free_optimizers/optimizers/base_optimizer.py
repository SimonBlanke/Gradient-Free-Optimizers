# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from .search_tracker import SearchTracker


class BaseOptimizer(SearchTracker):
    def __init__(self, search_space):
        super().__init__()
        self.search_space = search_space
        self.space_dim = np.array([array.size - 1 for array in search_space])

        self.eval_times = []
        self.iter_times = []

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
        self.pos_new = np.random.randint(self.space_dim, size=self.space_dim.shape)
        return self.pos_new

    def init_pos(self, pos):
        self.pos_new = pos

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()
