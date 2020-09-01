# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

np.seterr(divide="ignore", invalid="ignore")

from ..base_optimizer import BaseOptimizer
from ...search import Search


def skip_refit_75(i):
    if i <= 33:
        return 1
    return int((i - 33) ** 0.75)


def skip_refit_50(i):
    if i <= 33:
        return 1
    return int((i - 33) ** 0.5)


def skip_refit_25(i):
    if i <= 33:
        return 1
    return int((i - 33) ** 0.25)


def never_skip_refit(i):
    return 1


skip_retrain_ = {
    "many": skip_refit_75,
    "some": skip_refit_50,
    "few": skip_refit_25,
    "never": never_skip_refit,
}


class SBOM(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        max_sample_size=1000000,
        warm_start_smbo=None,
        skip_retrain="never",
    ):
        super().__init__(search_space)

        self.max_sample_size = max_sample_size
        self.warm_start_smbo = warm_start_smbo
        self.skip_retrain = skip_retrain_[skip_retrain]

        self.X_sample = []
        self.Y_sample = []

        self._all_possible_pos()

        if self.warm_start_smbo is not None:
            (self.X_sample, self.Y_sample) = self.warm_start_smbo

    def get_random_sample(self):
        sample_size = self._sample_size()
        if sample_size > self.all_pos_comb.shape[0]:
            sample_size = self.all_pos_comb.shape[0]

        row_sample = np.random.choice(
            self.all_pos_comb.shape[0], size=(sample_size,), replace=False
        )
        return self.all_pos_comb[row_sample]

    def _sample_size(self):
        n = self.max_sample_size
        return int(n * np.tanh(self.all_pos_comb.size / n))

    def _all_possible_pos(self):
        pos_space = []
        for dim_ in self.space_dim:
            pos_space.append(np.arange(dim_ + 1))

        self.n_dim = len(pos_space)
        self.all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, self.n_dim)

        # _split_into_subcubes(self.all_pos_comb)

    def init_pos(self, pos):
        super().init_pos(pos)
        self.X_sample.append(pos)

        return pos

