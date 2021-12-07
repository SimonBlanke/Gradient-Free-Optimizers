# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
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
    name = "Downhill Simplex Optimizer"

    def __init__(self, *args, alpha=1, gamma=2, beta=0.5, sigma=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma

        self.n_simp_positions = len(self.conv.search_space) + 1
        self.simp_positions = []

        self.simplex_step = 0

    def finish_initialization(self):
        idx_sorted = sort_list_idx(self.scores_valid)
        self.simplex_pos = [self.positions_valid[idx] for idx in idx_sorted]
        self.simplex_scores = [self.scores_valid[idx] for idx in idx_sorted]

        n_inits = len(self.positions_valid)
        if n_inits < self.n_simp_positions:
            print("\n Error: Not enough initial positions to form simplex")
            print("\n Increase number of initial positions")

        self.simplex_step = 1

        self.i_x_0 = 0
        self.i_x_N_1 = -2
        self.i_x_N = -1

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        simplex_stale = all(
            [np.array_equal(self.simplex_pos[0], array) for array in self.simplex_pos]
        )

        if simplex_stale:
            idx_sorted = sort_list_idx(self.scores_valid)
            self.simplex_pos = [self.positions_valid[idx] for idx in idx_sorted]
            self.simplex_scores = [self.scores_valid[idx] for idx in idx_sorted]

            self.simplex_step = 1

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
                self.center_array - self.simplex_pos[-1]
            )
            self.e_pos = self.conv2pos(e_pos)
            self.simplex_step = 1

            return self.e_pos

        elif self.simplex_step == 3:
            # iter Contraction
            c_pos = self.h_pos + self.beta * (self.center_array - self.h_pos)
            c_pos = self.conv2pos(c_pos)

            return c_pos

        elif self.simplex_step == 4:
            # iter Shrink
            pos = self.simplex_pos[self.compress_idx]
            pos = pos + self.sigma * (self.simplex_pos[0] - pos)

            return self.conv2pos(pos)

    def evaluate(self, score_new):
        self.score_new = score_new

        if self.simplex_step != 0:
            self.prev_pos = self.positions_valid[-1]

        if self.simplex_step == 1:
            # self.r_pos = self.prev_pos
            self.r_score = score_new

            if self.r_score > self.simplex_scores[0]:
                self.simplex_step = 2

            elif self.r_score > self.simplex_scores[-2]:
                # if r is better than x N-1
                self.simplex_pos[-1] = self.r_pos
                self.simplex_scores[-1] = self.r_score
                self.simplex_step = 1

            if self.simplex_scores[-1] > self.r_score:
                self.h_pos = self.simplex_pos[-1]
                self.h_score = self.simplex_scores[-1]
            else:
                self.h_pos = self.r_pos
                self.h_score = self.r_score

            self.simplex_step = 3

        elif self.simplex_step == 2:
            self.e_score = score_new

            if self.e_score > self.r_score:
                self.simplex_scores[-1] = self.e_pos
            elif self.r_score > self.e_score:
                self.simplex_scores[-1] = self.r_pos
            else:
                self.simplex_scores[-1] = random.choice([self.e_pos, self.r_pos])[0]

        elif self.simplex_step == 3:
            # eval Contraction
            self.c_pos = self.prev_pos
            self.c_score = score_new

            if self.c_score > self.simplex_scores[-1]:
                self.simplex_scores[-1] = self.c_score
                self.simplex_pos[-1] = self.c_pos

                self.simplex_step = 1

            else:
                # start Shrink
                self.simplex_step = 4
                self.compress_idx = 0

        elif self.simplex_step == 4:
            # eval Shrink
            self.simplex_scores[self.compress_idx] = score_new
            self.simplex_pos[self.compress_idx] = self.prev_pos

            self.compress_idx += 1

            if self.compress_idx == self.n_simp_positions:
                self.simplex_step = 1
