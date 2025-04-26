# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import scipy
import random
import numpy as np

from .search_tracker import SearchTracker
from .converter import Converter
from .init_positions import Initializer

from .utils import set_random_seed, move_random

from numpy.random import normal, laplace, logistic, gumbel

dist_dict = {
    "normal": normal,
    "laplace": laplace,
    "logistic": logistic,
    "gumbel": gumbel,
}


class CoreOptimizer(SearchTracker):
    def __init__(
        self,
        search_space,
        initialize,
        constraints,
        random_state,
        rand_rest_p,
        nth_process,
    ):
        super().__init__()

        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process

        self.random_seed = set_random_seed(self.nth_process, self.random_state)

        self.conv = Converter(self.search_space, self.constraints)
        self.init = Initializer(self.conv, self.initialize)

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

    def move_climb(
        self, pos, epsilon=0.03, distribution="normal", epsilon_mod=1
    ):
        while True:
            sigma = self.conv.max_positions * epsilon * epsilon_mod
            pos_normal = dist_dict[distribution](pos, sigma, pos.shape)
            pos = self.conv2pos(pos_normal)

            if self.conv.not_in_constraint(pos):
                return pos
            epsilon_mod *= 1.01

    def conv2pos(self, pos):
        # position to int
        r_pos = np.rint(pos)

        n_zeros = [0] * len(self.conv.max_positions)
        # clip into search space boundaries
        pos = np.clip(r_pos, n_zeros, self.conv.max_positions).astype(int)

        dist = scipy.spatial.distance.cdist(
            r_pos.reshape(1, -1), pos.reshape(1, -1)
        )
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
