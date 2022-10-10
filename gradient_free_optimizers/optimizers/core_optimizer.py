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


class CoreOptimizer(SearchTracker):
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

        self.random_seed = set_random_seed(nth_process, random_state)

        init = Initializer(self.conv)
        self.init_positions = init.set_pos(self.initialize)
        self.n_inits = get_n_inits(initialize)

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

    def move_random(self):
        return move_random(self.conv.search_space_positions)

    def add_n_random_init_pos(self, n):
        for _ in range(n):
            self.init_positions.append(self.move_random())

    def init_pos(self, pos):
        raise NotImplementedError

    def finish_initialization(self):
        raise NotImplementedError

    def evaluate(self, score_new):
        raise NotImplementedError
