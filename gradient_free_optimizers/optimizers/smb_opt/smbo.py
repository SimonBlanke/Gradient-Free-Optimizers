# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search
from .sampling import InitialSampler

import numpy as np

np.seterr(divide="ignore", invalid="ignore")


class SMBO(BaseOptimizer, Search):
    def __init__(
        self,
        *args,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        # warnings={"training": 100000, "prediction": 100000000},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.warm_start_smbo = warm_start_smbo
        self.sampling = sampling
        # self.warnings = warnings

        self.sampler = InitialSampler(self.conv, max_sample_size)

        # if self.warnings:
        #     self.memory_warning(max_sample_size)

        self.X_sample = []
        self.Y_sample = []

    def init_warm_start_smbo(self):
        if self.warm_start_smbo is not None:
            # filter out nan and inf
            warm_start_smbo = self.warm_start_smbo[
                ~self.warm_start_smbo.isin([np.nan, np.inf, -np.inf]).any(1)
            ]

            # filter out elements that are not in search space
            int_idx_list = []
            for para_name in self.conv.para_names:
                search_data_dim = warm_start_smbo[para_name].values
                search_space_dim = self.conv.search_space[para_name]

                int_idx = np.nonzero(np.in1d(search_data_dim, search_space_dim))[0]
                int_idx_list.append(int_idx)

            intersec = int_idx_list[0]
            for int_idx in int_idx_list[1:]:
                intersec = np.intersect1d(intersec, int_idx)
            warm_start_smbo_f = warm_start_smbo.iloc[intersec]

            X_sample_values = warm_start_smbo_f[self.conv.para_names].values
            Y_sample = warm_start_smbo_f["score"].values

            self.X_sample = self.conv.values2positions(X_sample_values)
            self.Y_sample = list(Y_sample)

    def track_X_sample(iterate):
        def wrapper(self, *args, **kwargs):
            pos = iterate(self, *args, **kwargs)
            self.X_sample.append(pos)
            return pos

        return wrapper

    def track_y_sample(evaluate):
        def wrapper(self, score):
            evaluate(self, score)

            if np.isnan(score) or np.isinf(score):
                del self.X_sample[-1]
            else:
                self.Y_sample.append(score)

        return wrapper

    def _sampling(self, all_pos_comb):
        if self.sampling is False:
            return all_pos_comb
        elif "random" in self.sampling:
            return self.random_sampling(all_pos_comb)

    def random_sampling(self, pos_comb):
        n_samples = self.sampling["random"]
        n_pos_comb = pos_comb.shape[0]

        if n_pos_comb <= n_samples:
            return pos_comb
        else:
            _idx_sample = np.random.choice(n_pos_comb, n_samples, replace=False)
            pos_comb_sampled = pos_comb[_idx_sample, :]
            return pos_comb_sampled

    def _all_possible_pos(self):
        pos_space = self.sampler.get_pos_space()
        # print("pos_space", pos_space)
        n_dim = len(pos_space)
        return np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

    def memory_warning(self, max_sample_size):
        if (
            self.conv.search_space_size > self.warnings
            and max_sample_size > self.warnings
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
        return super().init_pos(pos)
