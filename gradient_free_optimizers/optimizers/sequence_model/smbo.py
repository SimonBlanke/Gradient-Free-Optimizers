# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search

import numpy as np
from itertools import compress

np.seterr(divide="ignore", invalid="ignore")


class SMBO(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        warm_start_smbo=None,
        sampling={"random": 100000},
        warnings=100000000,
    ):
        super().__init__(search_space, initialize)
        self.warm_start_smbo = warm_start_smbo
        self.sampling = sampling
        self.warnings = warnings

    def init_position_combinations(self):
        search_space_size = 1
        for value_ in self.conv.search_space.values():
            search_space_size *= len(value_)

        self.X_sample = []
        self.Y_sample = []

        if self.warnings:
            self.memory_warning(search_space_size)
        self.all_pos_comb = self._all_possible_pos()

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

    def random_sampling(self):
        n_samples = self.sampling["random"]
        n_pos_comb = self.all_pos_comb.shape[0]

        if n_pos_comb <= n_samples:
            return self.all_pos_comb
        else:
            _idx_sample = np.random.choice(n_pos_comb, n_samples, replace=False)
            pos_comb_sampled = self.all_pos_comb[_idx_sample, :]
            return pos_comb_sampled

    def _all_possible_pos(self):
        if self.conv.max_dim < 255:
            _dtype = np.uint8
        elif self.conv.max_dim < 65535:
            _dtype = np.uint16
        elif self.conv.max_dim < 4294967295:
            _dtype = np.uint32
        else:
            _dtype = np.uint64

        pos_space = []
        for dim_ in self.conv.dim_sizes:
            pos_space.append(np.arange(dim_, dtype=_dtype))

        n_dim = len(pos_space)
        return np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

    def memory_warning(self, search_space_size):
        if search_space_size > self.warnings:
            warning_message0 = "\n Warning:"
            warning_message1 = (
                "\n search space size of "
                + str(search_space_size)
                + " exceeding recommended limit."
            )
            warning_message3 = "\n Reduce search space size for better performance."
            print(warning_message0 + warning_message1 + warning_message3)

    @track_X_sample
    def init_pos(self, pos):
        super().init_pos(pos)
        return pos
