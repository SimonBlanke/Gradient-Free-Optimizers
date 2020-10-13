# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
from .search_tracker import SearchTracker


class BaseOptimizer(SearchTracker):
    def __init__(self, search_space, rand_rest_p=0):
        super().__init__()
        self.search_space = search_space
        self.dim_sizes = np.array(
            [len(array) for array in search_space.values()]
        )
        self.max_positions = self.dim_sizes - 1
        self.rand_rest_p = rand_rest_p

        self.optimizers = [self]

    def move_random(self):
        self.pos_new = np.random.randint(
            self.dim_sizes, size=self.dim_sizes.shape
        )

        return self.pos_new

    def track_nth_iter(func):
        def wrapper(self, *args, **kwargs):
            self.nth_iter = len(self.score_new_list)
            return func(self, *args, **kwargs)

        return wrapper

    def random_restart(func):
        def wrapper(self, *args, **kwargs):
            if self.rand_rest_p > random.uniform(0, 1):
                return self.move_random()
            else:
                return func(self, *args, **kwargs)

        return wrapper

    @track_nth_iter
    def init_pos(self, pos):
        self.pos_new = pos

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()
