# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search
from ._sub_search_spaces import SubSearchSpaces
from ..smb_opt import BayesianOptimizer


class EvoSubSpaceBayesianOptimizer(BaseOptimizer, Search):
    name = "Subspace Bayesian Optimizer"

    def __init__(self, *args, max_size=300000, n_ss_min=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_ss_min = n_ss_min

        sub_search_spaces = SubSearchSpaces(self.conv.search_space, max_size=max_size)
        sub_search_spaces_l = sub_search_spaces.slice()

        self.sss_ids = list(range(len(sub_search_spaces_l)))
        self.sss_id_d = dict(zip(self.sss_ids, sub_search_spaces_l))
        self.sss_best_score_d = {}
        self.sss_scores_d = {}
        self.sss_bayes_opt_d = {}

        for sss_id in self.sss_ids:
            self.sss_best_score_d[sss_id] = -np.inf
            self.sss_scores_d[sss_id] = []

            sub_search_space = self.sss_id_d[sss_id]
            self.sss_bayes_opt_d[sss_id] = BayesianOptimizer(
                sub_search_space, self.initialize
            )

        self.n_sub_spaces = len(sub_search_spaces_l)
        print("\n self.n_sub_spaces \n", self.n_sub_spaces)

        self.n_iter_evo = sum(self.initialize.values()) * self.n_sub_spaces * 2
        print("\n self.n_iter_evo \n", self.n_iter_evo)

    def init_pos(self, pos):
        self.pos_new = pos
        self.nth_iter = len(self.pos_new_list)
        return self.iterate()

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        self.sss_id = random.choice(self.sss_ids)
        bayes_opt = self.sss_bayes_opt_d[self.sss_id]
        # print("\n\n\n self.sss_id ", self.sss_id, "\n")

        if bayes_opt.init_positions:
            init_pos = bayes_opt.init_positions[0]
            pos_new = bayes_opt.init_pos(init_pos)
            bayes_opt.init_positions = bayes_opt.init_positions[1:]
        else:
            pos_new = bayes_opt.iterate()

        sub_search_space = self.sss_id_d[self.sss_id]
        # print("\n sub_search_space \n", sub_search_space)
        pos_sss = bayes_opt.iterate()
        value_sss = bayes_opt.conv.position2value(pos_sss)
        pos_new = self.conv.value2position(value_sss)
        # print("\n n sss_id \n", len(self.sss_ids))

        return pos_new

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

        self.sss_scores_d[self.sss_id].append(score_new)
        if score_new > self.sss_best_score_d[self.sss_id]:
            self.sss_best_score_d[self.sss_id] = score_new

        # evolutionary step
        modZero = self.nth_iter % self.n_iter_evo == 0
        if modZero and len(self.sss_ids) > self.n_ss_min:
            sss_id_worst = None
            metric_worst = np.inf

            for sss_id in self.sss_ids:
                scores = np.array(self.sss_scores_d[sss_id])
                # metric = scores.mean()
                metric = np.amax(scores)
                if metric < metric_worst:
                    metric_worst = metric
                    sss_id_worst = sss_id

            self.sss_ids.remove(sss_id_worst)
