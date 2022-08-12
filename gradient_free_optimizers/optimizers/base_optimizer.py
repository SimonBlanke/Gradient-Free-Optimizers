# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import scipy
import random
import numpy as np
from .search_tracker import SearchTracker
from ..converter import Converter
from ..results_manager import ResultsManager
from ..init_positions import Initializer

from ..utils import set_random_seed, move_random


def get_n_inits(initialize):
    n_inits = 0
    for key_ in initialize.keys():
        init_value = initialize[key_]
        if isinstance(init_value, int):
            n_inits += init_value
        else:
            n_inits += len(init_value)
    return n_inits


class BaseOptimizer(SearchTracker):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__()
        self.conv = Converter(search_space)
        self.results_mang = ResultsManager(self.conv)
        self.initialize = initialize
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process

        self.state = "init"

        self.optimizers = [self]

        self.random_seed = set_random_seed(nth_process, random_state)

        # get init positions
        init = Initializer(self.conv)
        self.init_positions = init.set_pos(self.initialize)

        self.n_inits = get_n_inits(initialize)

    def move_random(self):
        return move_random(self.conv.search_space_positions)

    def random_iteration(func):
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

        dist = scipy.spatial.distance.cdist(r_pos.reshape(1, -1), pos.reshape(1, -1))
        threshold = self.conv.search_space_size / (100 ** self.conv.n_dimensions)

        if dist > threshold:
            return self.move_random()

        return pos

    def init_pos(self, pos):
        self.pos_new = pos
        return pos

    def finish_initialization(self):
        pass

    def evaluate(self, score_new):
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.pos_current = self.pos_new

            self.score_best = score_new
            self.score_current = score_new

        # self._evaluate_new2current(score_new)
        # self._evaluate_current2best()
