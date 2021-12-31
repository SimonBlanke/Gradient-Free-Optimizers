# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search
from ._sub_search_spaces import SubSearchSpaces
from ..smb_opt import BayesianOptimizer
from ..smb_opt.smbo import SMBO


class VariableResolutionBayesianOptimizer(SMBO):
    name = "Variable Resolution Bayesian Optimizer"

    def __init__(self, *args, n_best_pos=5, n_iter_reso=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_best_pos = n_best_pos
        self.n_iter_reso = n_iter_reso

        self.score_para_d = {}
        self.search_space_reso = self.conv.search_space
        self.bayes = BayesianOptimizer(self.conv.search_space)

    @SMBO.track_nth_iter
    @SMBO.track_X_sample
    def iterate(self):
        modZero = self.nth_iter % self.n_iter_reso == 0

        if modZero or self.nth_iter == 0:
            self.search_space_reso = self.decrease_ss_reso(
                self.conv.search_space, para_list=list(self.score_para_d.values())
            )
            self.bayes = BayesianOptimizer(self.search_space_reso, initialize={})
            self.bayes.X_sample = self.X_sample
            self.bayes.Y_sample = self.Y_sample

        pos_reso = self.bayes.iterate()
        value = self.bayes.conv.position2value(pos_reso)
        pos_new = self.conv.value2position(value)
        return pos_new

    @SMBO.track_y_sample
    def evaluate(self, score_new):
        self.bayes.evaluate(score_new)

        value = self.conv.position2value(self.pos_new_list[-1])
        para = self.conv.value2para(value)

        self.score_para_d[score_new] = para

        # get best scores from score_para_d
        bestn_scores = sorted(list(self.score_para_d.keys()))[-self.n_best_pos :]

        score_para_d_tmp = {}
        for score, para in self.score_para_d.items():
            if score in bestn_scores:
                score_para_d_tmp[score] = para

        self.score_para_d = score_para_d_tmp

    def decrease_ss_reso(
        self, search_space, para_list=None, min_dim_size=20, margin=5, n_samples=20
    ):
        search_space_reso = {}
        # para_list = None

        for para in list(search_space.keys()):
            dim_values = search_space[para]
            dim_size = len(dim_values)

            if dim_size > min_dim_size:
                f_reso = int(dim_size / min_dim_size)
                dim_values_new = dim_values[::f_reso]

                if para_list is None or len(para_list) == 0:
                    search_space_reso[para] = dim_values_new
                else:
                    dim_values_cen_l = [dim_values_new]
                    for para_d in para_list:
                        density = (max(dim_values) - min(dim_values)) / len(dim_values)

                        center = para_d[para]
                        min_ = center - density * dim_size * margin / 100
                        max_ = center + density * dim_size * margin / 100

                        dim_pos_center = np.where(
                            np.logical_and(dim_values >= min_, dim_values <= max_)
                        )[0]
                        dim_values_center = dim_values[dim_pos_center]
                        samples = np.random.choice(dim_values_center, size=n_samples)

                        dim_values_cen_l.append(samples)

                    dim_values_conc = np.hstack((dim_values_cen_l))
                    dim_values_unique = np.unique(dim_values_conc)
                    dim_values_sort = np.sort(dim_values_unique)

                    search_space_reso[para] = dim_values_sort
            else:
                search_space_reso[para] = dim_values

        return search_space_reso
