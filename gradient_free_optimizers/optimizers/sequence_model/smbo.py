# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

np.seterr(divide="ignore", invalid="ignore")

from ..base_optimizer import BaseOptimizer
from ...search import Search


class SMBO(BaseOptimizer, Search):
    def __init__(
        self, search_space, warm_start_smbo=None,
    ):
        super().__init__(search_space)
        self.warm_start_smbo = warm_start_smbo

        self.X_sample = []
        self.Y_sample = []

        if self.warm_start_smbo is not None:
            (X_sample_values, Y_sample) = self.warm_start_smbo

            self.X_sample = self.conv.values2positions(X_sample_values)
            self.Y_sample = Y_sample

        self.all_pos_comb = self._all_possible_pos()

    def track_X_sample(func):
        def wrapper(self, *args, **kwargs):
            pos = func(self, *args, **kwargs)
            self.X_sample.append(pos)
            return pos

        return wrapper

    def _all_possible_pos(self):
        pos_space = []
        for dim_ in self.conv.max_positions:
            pos_space.append(np.arange(dim_))

        n_dim = len(pos_space)
        return np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

    @track_X_sample
    def init_pos(self, pos):
        super().init_pos(pos)
        return pos

