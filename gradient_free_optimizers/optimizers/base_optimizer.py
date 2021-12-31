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


def set_random_seed(nth_process, random_state):
    """
    Sets the random seed separately for each thread
    (to avoid getting the same results in each thread)
    """
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = np.random.randint(0, high=2 ** 31 - 2, dtype=np.int64)

    random.seed(random_state + nth_process)
    np.random.seed(random_state + nth_process)


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

        set_random_seed(nth_process, random_state)

        # get init positions
        init = Initializer(self.conv)
        self.init_positions = init.set_pos(self.initialize)

        self.n_inits = get_n_inits(initialize)

    def move_random(self):
        return move_random(self.conv.search_space_positions)

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

        dist = scipy.spatial.distance.cdist(r_pos.reshape(1, -1), pos.reshape(1, -1))

        if dist > self.conv.search_space_size / 1000000:
            return self.move_random()

        return pos

    def init_pos(self, pos):
        self.pos_new = pos
        self.nth_iter = len(self.pos_new_list)
        return pos

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
