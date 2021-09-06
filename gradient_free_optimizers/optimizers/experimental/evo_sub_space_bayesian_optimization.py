# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search
from ._sub_search_spaces import SubSearchSpaces
from ..sequence_model import BayesianOptimizer


class EvoSubSpaceBayesianOptimizer(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        random_state=None,
        max_size=300000,
    ):
        super().__init__(search_space, initialize, random_state)

        sub_search_spaces = SubSearchSpaces(search_space, max_size=max_size)

        self.sss_ids = range(len(sub_search_spaces))
        self.sss_id_d = dict(zip(self.sss_ids, sub_search_spaces))
        self.sss_best_score_d = {}
        self.sss_bayes_opt_d = {}

        for sss_id in self.sss_ids:
            self.sss_best_score_d[sss_id] = -np.inf

            sub_search_space = self.sss_id_d[sss_id]
            self.sss_bayes_opt_d[sss_id] = BayesianOptimizer(
                sub_search_space, initialize
            )

        self.n_sub_spaces = len(self.sub_search_spaces)

    def init_pos(self, pos):
        self.pos_new = pos
        self.nth_iter = len(self.pos_new_list)
        return pos

    def finish_initialization(self):
        pass

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        sss_id = random.choice(self.sss_ids)
        bayes_opt = self.sss_bayes_opt_d[sss_id]

        if len(bayes_opt.X_sample) < self.init_positions:
            pos_new = bayes_opt.iterate()

        else:
            pos_new = bayes_opt.iterate()

        return pos_new

    def evaluate(self, score_new):
        pass
