# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from .search_tracker import SearchTracker


class BaseOptimizer(SearchTracker):
    def __init__(self, search_space):
        super().__init__()
        self.search_space = search_space
        self.space_dim_size = np.array([array.size for array in search_space])

        self.optimizers = [self]

    def move_random(self):
        self.pos_new = np.random.randint(
            self.space_dim_size, size=self.space_dim_size.shape
        )
        return self.pos_new

    def iter_dec(func):
        def wrapper(self, *args, **kwargs):
            self.nth_iter = len(self.score_new_list)
            return func(self, *args, **kwargs)

        return wrapper

    @iter_dec
    def init_pos(self, pos):
        self.pos_new = pos

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()
