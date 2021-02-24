# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from itertools import compress

np.seterr(divide="ignore", invalid="ignore")

from ..base_optimizer import BaseOptimizer
from ...search import Search


def memory_warning_1(search_space_size):
    if search_space_size > 10000000:
        warning_message0 = "\n Warning:"
        warning_message1 = "\n search space too large for smb-optimization."
        warning_message3 = "\n Reduce search space size for better performance."
        print(warning_message0 + warning_message1 + warning_message3)


def memory_warning_2(all_pos_comb):
    all_pos_comb_gbyte = all_pos_comb.nbytes / 3000000000
    if all_pos_comb_gbyte > 1:
        warning_message0 = "\n Warning:"
        warning_message2 = "\n Memory-load exceeding recommended limit."
        print(warning_message0 + warning_message2)


class SMBO(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        warm_start_smbo=None,
    ):
        super().__init__(search_space, initialize)
        self.warm_start_smbo = warm_start_smbo

        search_space_size = 1
        for value_ in search_space.values():
            search_space_size *= len(value_)

        self.X_sample = []
        self.Y_sample = []

        memory_warning_1(search_space_size)
        self.all_pos_comb = self._all_possible_pos()
        memory_warning_2(self.all_pos_comb)

    def init_warm_start_smbo(self):
        if self.warm_start_smbo is not None:
            X_sample_values = self.warm_start_smbo[self.conv.para_names].values
            Y_sample = self.warm_start_smbo["score"].values

            self.X_sample = self.conv.values2positions(X_sample_values)
            self.Y_sample = list(Y_sample)

            # filter out nan
            mask = ~np.isnan(Y_sample)
            self.X_sample = list(compress(self.X_sample, mask))
            self.Y_sample = list(compress(self.Y_sample, mask))

    def track_X_sample(func):
        def wrapper(self, *args, **kwargs):
            pos = func(self, *args, **kwargs)
            self.X_sample.append(pos)
            return pos

        return wrapper

    def _all_possible_pos(self):
        pos_space = []
        for dim_ in self.conv.dim_sizes:
            pos_space.append(np.arange(dim_))

        n_dim = len(pos_space)
        return np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

    @track_X_sample
    def init_pos(self, pos):
        super().init_pos(pos)
        return pos
