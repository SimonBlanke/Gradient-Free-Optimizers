# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random
import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search
from ._sub_search_spaces import SubSearchSpaces
from ..smb_opt import BayesianOptimizer


class LocalBayesianOptimizer(BaseOptimizer, Search):
    name = "Local Bayesian Optimizer"

    def __init__(
        self, *args, max_size=300000, n_positions=20, local_range=100, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.max_size = max_size
        self.n_positions = n_positions
        self.local_range = local_range

        self.bayes_opt = BayesianOptimizer(self.conv.search_space)

    def create_local_smbo(self, current_position):
        local_ss = {}

        for idx, para in enumerate(self.conv.para_names):

            max_dim = max(0, current_position[idx] + self.local_range)
            min_dim = min(
                self.conv.dim_sizes[idx], current_position[idx] - self.local_range
            )

            dim_pos = np.array(self.conv.search_space_positions[idx])

            dim_pos_center = np.where(
                np.logical_and(dim_pos >= min_dim, dim_pos <= max_dim)
            )[0]
            local_ss[para] = dim_pos_center

        self.bayes_opt = BayesianOptimizer(local_ss)

    def finish_initialization(self):
        self.create_local_smbo(self.pos_current)

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        pos_loc = self.bayes_opt.iterate()
        pos_new = self.bayes_opt.conv.position2value(pos_loc)

        return pos_new

    def evaluate(self, score_new):
        self.bayes_opt.evaluate(score_new)

        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

        modZero = self.nth_iter % self.n_positions == 0

        if modZero:
            self.create_local_smbo(self.pos_current)
