# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class BasePositioner:
    def __init__(self, space_dim, _opt_args_):
        self.space_dim = space_dim

        self.epsilon = _opt_args_.epsilon
        self.distribution = _opt_args_.distribution

        self._pos_new = None
        self._score_new = -np.inf

        self._pos_current = None
        self._score_current = -np.inf

        self.pos_best = None
        self.score_best = -np.inf

        self.pos_list = []
        self.score_list = []

        self.pos_current_list = []
        self.score_current_list = []

    @property
    def pos_new(self):
        return self._pos_new

    @pos_new.setter
    def pos_new(self, value):
        self.pos_list.append(value)
        self._pos_new = value

    @property
    def score_new(self):
        return self._score_new

    @score_new.setter
    def score_new(self, value):
        self.score_list.append(value)
        self._score_new = value

    @property
    def score_current(self):
        return self._score_current

    @score_current.setter
    def score_current(self, value):
        self.score_current_list.append(value)
        self._score_current = value

    @property
    def pos_current(self):
        return self._pos_current

    @pos_current.setter
    def pos_current(self, value):
        self.pos_current_list.append(value)
        self._pos_current = value

    def move_climb(self, pos, epsilon_mod=1):
        sigma = self.space_dim * self.epsilon * epsilon_mod
        pos_normal = self.distribution(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(self.space_dim)
        pos = np.clip(pos_new_int, n_zeros, self.space_dim)

        self.pos_new = pos.astype(int)
        return self.pos_new

    def move_random(self):
        pos_new = np.random.uniform(
            np.zeros(self.space_dim.shape), self.space_dim, self.space_dim.shape
        )
        self.pos_new = np.rint(pos_new).astype(int)
        return self.pos_new
