# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from .opt_args import Arguments


class BaseOptimizer:
    def __init__(self, init_positions, space_dim, opt_para):
        self._opt_args_ = Arguments(**opt_para)
        self._opt_args_.set_opt_args()

        self.init_positions = init_positions
        self.space_dim = space_dim
        self.opt_para = opt_para

        self.nth_iter = 0
        self.p_list = []

    def _base_init_pos(self, nth_init, positioner):
        init_position = self.init_positions[nth_init]

        self.p_current = positioner
        self.p_current.pos_new = init_position

        self.p_list.append(self.p_current)

        return init_position

    def _sort_(self):
        self.p_sorted = self.p_list

    def _sort_best(self):
        scores_list = []
        for _p_ in self.p_list:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        self.p_sorted = [self.p_list[i] for i in idx_sorted_ind]

    def _base_iterate(self, nth_iter):
        self.nth_iter = nth_iter

    def _choose_next_pos(self):
        self.p_current = self.p_list[self.nth_iter % len(self.p_list)]
        self.p_rest = [p for p in self.p_list if p is not self.p_current]

    def _evaluate_new2current(self, score_new):
        if score_new >= self.p_current.score_current:
            self.p_current.score_current = score_new
            self.p_current.pos_current = self.p_current.pos_new

    def _evaluate_current2best(self):
        if self.p_current.score_current >= self.p_current.score_best:
            self.p_current.score_best = self.p_current.score_current
            self.p_current.pos_best = self.p_current.pos_current

    def _current2best(self):
        self.p_current.score_best = self.p_current.score_current
        self.p_current.pos_best = self.p_current.pos_current

    def _new2current(self):
        self.p_current.score_current = self.p_current.score_new
        self.p_current.pos_current = self.p_current.pos_new
