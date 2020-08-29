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

        self.optimizers = [self]

    def move_random(self):
        self.pos_new = np.random.randint(self.space_dim, size=self.space_dim.shape)
        return self.pos_new

    def init_pos(self, pos):
        self.pos_new = pos

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()
