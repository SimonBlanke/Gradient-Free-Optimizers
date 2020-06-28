# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np


from ..base_optimizer import BaseOptimizer
from ..base_positioner import BasePositioner


def _split_into_subcubes(data, split_per_dim=2):
    n_dim = data.shape[1]
    subcubes = []

    data_list = [data]

    for dim in range(n_dim):
        subdata_list = []

        if dim == 0:
            data_list = [data]

        for data in data_list:
            data_sorted = data[data[:, dim].argsort()]

            subdata = np.array_split(data_sorted, 2, axis=0)
            subdata_list = subdata_list + subdata

        data_list = subdata_list

    return subcubes


class SBOM(BaseOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.X_sample = []
        self.Y_sample = []

    def get_random_sample(self):
        sample_size = self._sample_size()
        if sample_size > self.all_pos_comb.shape[0]:
            sample_size = self.all_pos_comb.shape[0]

        row_sample = np.random.choice(
            self.all_pos_comb.shape[0], size=(sample_size,), replace=False
        )
        return self.all_pos_comb[row_sample]

    def _sample_size(self):
        n = self._opt_args_.max_sample_size
        return int(n * np.tanh(self.all_pos_comb.size / n))

    def _all_possible_pos(self):
        pos_space = []
        for dim_ in self.space_dim:
            pos_space.append(np.arange(dim_ + 1))

        self.n_dim = len(pos_space)
        self.all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, self.n_dim)

        # _split_into_subcubes(self.all_pos_comb)

    def init_pos(self, nth_init):
        pos_new = self._base_init_pos(
            nth_init, SbomPositioner(self.space_dim, self._opt_args_)
        )

        self._all_possible_pos()

        if self._opt_args_.warm_start_smbo is not None:
            (self.X_sample, self.Y_sample) = self._opt_args_.warm_start_smbo

        self.X_sample.append(pos_new)

        return pos_new


class SbomPositioner(BasePositioner):
    def __init__(self, space_dim, opt_para):
        super().__init__(space_dim, opt_para)
