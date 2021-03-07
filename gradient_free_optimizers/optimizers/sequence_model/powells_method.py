# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search
from .bayesian_optimization import BayesianOptimizer


def sort_list_idx(list_):
    list_np = np.array(list_)
    idx_sorted = list(list_np.argsort()[::-1])
    return idx_sorted


class PowellsMethod(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        iters_p_dim=20,
    ):
        super().__init__(search_space, initialize)

        self.iters_p_dim = iters_p_dim

        self.current_search_dim = -1

    def finish_initialization(self):
        self.nth_iter_ = -1
        self.nth_iter_current_dim = 0

    def new_dim(self):
        self.current_search_dim += 1

        if self.current_search_dim >= self.conv.n_dimensions:
            self.current_search_dim = 0

        idx_sorted = sort_list_idx(self.scores_valid)
        self.powells_pos = [self.positions_valid[idx] for idx in idx_sorted][0]
        self.powells_scores = [self.scores_valid[idx] for idx in idx_sorted][0]

        self.nth_iter_current_dim = 0

        min_pos = []
        max_pos = []
        center_pos = []

        search_space_1D = {}
        for idx, para_name in enumerate(self.conv.para_names):
            if self.current_search_dim == idx:
                # fill with range of values
                search_space_pos = self.conv.search_space_positions[idx]
                search_space_1D[para_name] = search_space_pos

                min_pos.append(int(np.amin(search_space_pos)))
                max_pos.append(int(np.amax(search_space_pos)))
                center_pos.append(int(np.median(search_space_pos)))
            else:
                # fill with single value
                search_space_1D[para_name] = np.array([self.powells_pos[idx]])

                min_pos.append(self.powells_pos[idx])
                max_pos.append(self.powells_pos[idx])
                center_pos.append(self.powells_pos[idx])

        self.init_positions_ = [min_pos, center_pos, max_pos]

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
            pos_new = self.bayes_opt.iterate()
            pos_new = self.bayes_opt.conv.position2value(pos_new)

        return pos_new

    def evaluate(self, score_new):
        self.score_new = score_new

        if self.current_search_dim == -1:
            BaseOptimizer.evaluate(self, score_new)
        else:
            self.bayes_opt.evaluate(score_new)
