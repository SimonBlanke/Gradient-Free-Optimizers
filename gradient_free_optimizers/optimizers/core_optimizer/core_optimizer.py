import scipy
import random
import numpy as np

from .search_tracker import SearchTracker
from .converter import Converter
from .init_positions import Initializer

from ...utils import set_random_seed, move_random


class CoreOptimizer(SearchTracker):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        debug_log=False,
    ):
        super().__init__()

        self.random_seed = set_random_seed(nth_process, random_state)

        self.conv = Converter(search_space, constraints)
        self.init = Initializer(self.conv, initialize)

        self.initialize = initialize
        self.constraints = constraints
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process
        self.debug_log = debug_log

        if self.constraints is None:
            self.constraints = []

        self.nth_init = 0
        self.nth_trial = 0
        self.search_state = "init"

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
        threshold = self.conv.search_space_size / (100**self.conv.n_dimensions)

        if dist > threshold:
            return self.move_random()

        return pos

    def move_random(self):
        while True:
            pos = move_random(self.conv.search_space_positions)
            if self.conv.not_in_constraint(pos):
                return pos

    @SearchTracker.track_new_pos
    def init_pos(self):
        init_pos = self.init.init_positions_l[self.nth_init]
        return init_pos

    @SearchTracker.track_new_score
    def evaluate_init(self, score_new):
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.score_best = score_new

        if self.pos_current is None:
            self.pos_current = self.pos_new
            self.score_current = score_new
