# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search
from ..smb_opt.bayesian_optimization import BayesianOptimizer


def sort_list_idx(list_):
    list_np = np.array(list_)
    idx_sorted = list(list_np.argsort()[::-1])
    return idx_sorted


class OneDimensionalBayesianOptimization(BaseOptimizer, Search):
    name = "One Dimensional Bayesian Optimizer"

    def __init__(self, *args, iters_p_dim=15, **kwargs):
        super().__init__(*args, **kwargs)

        self.iters_p_dim = iters_p_dim

        self.current_search_dim = -1

    def finish_initialization(self):
        self.nth_iter_ = -1
        self.nth_iter_current_dim = 0

    def new_dim(self):
        self.current_search_dim += 1

        if self.current_search_dim >= self.conv.n_dimensions:
            self.current_search_dim = 0

        last_n_pos = self.positions_valid[-self.iters_p_dim :]
        last_n_scores = self.scores_valid[-self.iters_p_dim :]

        idx_sorted = sort_list_idx(last_n_scores)
        self.powells_pos = [last_n_pos[idx] for idx in idx_sorted][0]
        self.powells_scores = [last_n_scores[idx] for idx in idx_sorted][0]

        self.nth_iter_current_dim = 0

        search_space_1D = {}
        for idx, para_name in enumerate(self.conv.para_names):
            if self.current_search_dim == idx:
                # fill with range of values
                search_space_1D[para_name] = self.conv.search_space[para_name]

            else:
                # fill with single value
                pow_value = self.conv.position2value(self.powells_pos)
                search_space_1D[para_name] = np.array([pow_value[idx]])

        self.bayes_opt = BayesianOptimizer(
            search_space=search_space_1D, initialize={"vertices": 2, "random": 3}
        )

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        self.nth_iter_ += 1
        self.nth_iter_current_dim += 1

        modZero = self.nth_iter_ % self.iters_p_dim == 0
        # nonZero = self.nth_iter_ != 0

        if modZero:
            self.new_dim()

        if self.nth_iter_current_dim < 5:
            pos_new = self.bayes_opt.init_pos(
                self.bayes_opt.init_positions[self.nth_iter_current_dim]
            )
        else:
            pos_one_dim = self.bayes_opt.iterate()
            value_one_dim = self.bayes_opt.conv.position2value(pos_one_dim)
            pos_new = self.conv.value2position(value_one_dim)

        return pos_new

    def evaluate(self, score_new):
        if self.current_search_dim == -1:
            BaseOptimizer.evaluate(self, score_new)
        else:
            self.score_new = score_new
            self.bayes_opt.evaluate(score_new)
