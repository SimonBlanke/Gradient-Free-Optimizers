# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search


def sort_list_idx(list_):
    list_np = np.array(list_)
    idx_sorted = list(list_np.argsort()[::-1])
    return idx_sorted


def centeroid(array_list):
    centeroid = []
    for idx in range(array_list[0].shape[0]):
        center_dim_pos = []
        for array in array_list:
            center_dim_pos.append(array[idx])

        center_dim_mean = np.array(center_dim_pos).mean()
        centeroid.append(center_dim_mean)

    return centeroid


class DownhillSimplexOptimizer(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        alpha=1,
        gamma=2,
        beta=0.5,
        sigma=0.5,
        rand_rest_p=0.01,
    ):
        super().__init__(search_space, initialize)

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma

        self.n_simp_positions = len(search_space) + 1
        self.simp_positions = []

        self.simplex_step = 0

    def finish_initialization(self):
        idx_sorted = sort_list_idx(self.scores_valid)
        self.simplex_pos = [self.positions_valid[idx] for idx in idx_sorted]
        self.simplex_scores = [self.scores_valid[idx] for idx in idx_sorted]

        self.simplex_step = 1

        self.i_x_0 = 0
        self.i_x_N_1 = -2
        self.i_x_N = -1

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        if self.simplex_step == 1:
            idx_sorted = sort_list_idx(self.simplex_scores)
            self.simplex_pos = [self.simplex_pos[idx] for idx in idx_sorted]
            self.simplex_scores = [self.simplex_scores[idx] for idx in idx_sorted]

            self.center_array = centeroid(self.simplex_pos[:-1])

            r_pos = self.center_array + self.alpha * (
                self.center_array - self.simplex_pos[-1]
            )

            self.r_pos = self.conv2pos(r_pos)
            return self.r_pos

        elif self.simplex_step == 2:
            e_pos = self.center_array + self.gamma * (
                self.center_array + self.simplex_pos[-1]
            )
            self.e_pos = self.conv2pos(e_pos)
            self.simplex_step = 1

            return self.e_pos

        elif self.simplex_step == 3:
            return self.r_pos

        elif self.simplex_step == 4:
            return self.c_pos

        elif self.simplex_step == 5:
            pos = self.simplex_pos[self.compress_idx]
            pos = pos + self.sigma * (self.simplex_scores[0] - pos)

            return self.conv2pos(pos)

    def evaluate(self, score_new):
        self.score_new = score_new

        if self.simplex_step == 1:
            if score_new > self.simplex_scores[0]:
                # if r is better than x0
                self.simplex_pos[-1] = self.r_pos
                self.simplex_scores[-1] = score_new
                self.simplex_step = 2
            elif score_new > self.simplex_scores[-2]:
                # if r is better than x N-1
                self.simplex_pos[-2] = self.r_pos
                self.simplex_scores[-2] = score_new
                self.simplex_step = 3
            elif score_new > self.simplex_scores[-1]:
                # if r is better than x N
                self.h_pos = self.r_pos
                c_pos = self.h_pos + self.beta * (self.center_array - self.h_pos)
                self.c_pos = self.conv2pos(c_pos)

                self.simplex_step = 4
            else:
                # if r is worse than x N
                self.h_pos = self.simplex_pos[-1]
                c_pos = self.h_pos + self.beta * (self.center_array - self.h_pos)
                self.c_pos = self.conv2pos(c_pos)

                self.simplex_step = 4

        elif self.simplex_step == 2:
            idx_sorted = sort_list_idx(self.scores_valid[-2:])
            self.simplex_pos[-1] = self.simplex_pos[-2:][idx_sorted][0]
            self.simplex_scores[-1] = self.simplex_scores[-2:][idx_sorted][0]

            self.simplex_step -= 1

        elif self.simplex_step == 3:
            self.simplex_step = 1

        elif self.simplex_step == 4:
            if score_new > self.simplex_scores[-1]:
                self.simplex_pos[-1] = self.c_pos
                self.simplex_scores[-1] = score_new
                self.simplex_step = 1
            else:
                self.simplex_step = 5
                self.compress_idx = 0

        elif self.simplex_step == 5:
            self.simplex_scores[self.compress_idx] = score_new
            self.compress_idx += 1

            if self.compress_idx == self.n_simp_positions:
                self.simplex_step = 1
