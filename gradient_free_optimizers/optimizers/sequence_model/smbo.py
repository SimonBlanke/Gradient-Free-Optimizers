# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search
from .sampling import InitialSampler

import numpy as np
from itertools import compress

np.seterr(divide="ignore", invalid="ignore")


class SMBO(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        warm_start_smbo=None,
        init_sample_size=10000000,
        sampling={"random": 1000000},
        warnings=100000000,
    ):
        super().__init__(search_space, initialize)
        self.warm_start_smbo = warm_start_smbo
        self.sampling = sampling
        self.warnings = warnings

        self.sampler = InitialSampler(self.conv, init_sample_size)

        if self.warnings:
            self.memory_warning(init_sample_size)

    def init_position_combinations(self):
        self.X_sample = []
        self.Y_sample = []

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
        pos_space = self.sampler.get_pos_space()
        # print("pos_space", pos_space)
        n_dim = len(pos_space)
        return np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

    def memory_warning(self, init_sample_size):
        if (
            self.conv.search_space_size > self.warnings
            and init_sample_size > self.warnings
        ):
            warning_message0 = "\n Warning:"
            warning_message1 = (
                "\n search space size of "
                + str(self.conv.search_space_size)
                + " exceeding recommended limit."
            )
            warning_message3 = "\n Reduce search space size for better performance."
            print(warning_message0 + warning_message1 + warning_message3)

    @track_X_sample
    def init_pos(self, pos):
        super().init_pos(pos)
        return pos
