# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
from .search_tracker import SearchTracker
from ..converter import Converter
from ..results_manager import ResultsManager


class BaseOptimizer(SearchTracker):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
    ):
        super().__init__()
        self.conv = Converter(search_space)
        self.results_mang = ResultsManager(self.conv)
        self.initialize = initialize

        self.optimizers = [self]

        n_inits = 0
        for key_ in self.initialize.keys():
            init_value = self.initialize[key_]
            if isinstance(init_value, int):
                n_inits += init_value
            else:
                n_inits += len(init_value)
        self.n_inits = n_inits

    def move_random(self):
        position = []
        for search_space_pos in self.conv.search_space_positions:
            pos_ = random.choice(search_space_pos)
            position.append(pos_)

        return np.array(position)

    def track_nth_iter(func):
        def wrapper(self, *args, **kwargs):
            self.nth_iter = len(self.pos_new_list)
            pos = func(self, *args, **kwargs)
            self.pos_new = pos
            return pos

        return wrapper

    def random_restart(func):
        def wrapper(self, *args, **kwargs):
            if self.rand_rest_p > random.uniform(0, 1):
                return self.move_random()
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def conv2pos(self, pos):
        # position to int
        r_pos = np.rint(pos)

        n_zeros = [0] * len(self.conv.max_positions)
        # clip into search space boundaries
        pos = np.clip(r_pos, n_zeros, self.conv.max_positions).astype(int)

        return pos

    def init_pos(self, pos):
        self.pos_new = pos
        self.nth_iter = len(self.pos_new_list)

    def finish_initialization(self):
        pass

    def evaluate(self, score_new):
        self.score_new = score_new

        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.pos_current = self.pos_new

            self.score_best = score_new
            self.score_current = score_new

        # self._evaluate_new2current(score_new)
        # self._evaluate_current2best()
