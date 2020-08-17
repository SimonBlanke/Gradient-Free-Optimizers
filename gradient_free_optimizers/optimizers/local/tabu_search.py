# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


from . import HillClimbingOptimizer
from scipy.spatial.distance import euclidean


def gaussian(distance, sig, sigma_factor=1):
    return (
        sigma_factor
        * sig
        * np.exp(-np.power(distance, 2.0) / (sigma_factor * np.power(sig, 2.0)))
    )


class TabuOptimizer(HillClimbingOptimizer):
    def __init__(self, space_dim, tabu_memory=3):
        super().__init__(space_dim)

        self.tabus = []
        self.tabu_memory = tabu_memory

    def add_tabu(self, tabu):
        self.tabus.append(tabu)

        if len(self.tabus) > self.tabu_memory:
            self.tabus.pop(0)

    def _move_climb(self, pos, epsilon_mod=1):
        sigma = 1 + self.space_dim * self.epsilon * epsilon_mod
        pos_normal = np.random.normal(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        sigma_mod = 1
        run = True
        while run:
            pos_normal = np.random.normal(pos, sigma * sigma_mod, pos.shape)
            pos_new_int = np.rint(pos_normal)

            p_discard_sum = []
            for tabu in self.tabus:
                distance = euclidean(pos_new_int, tabu)
                sigma_mean = sigma.mean()
                p_discard = gaussian(distance, sigma_mean)

                p_discard_sum.append(p_discard)

            p_discard = np.array(p_discard_sum).sum()
            rand = random.uniform(0, 1)

            if p_discard < rand:
                run = False

            sigma_mod = sigma_mod * 1.01

        n_zeros = [0] * len(self.space_dim)
        pos = np.clip(pos_new_int, n_zeros, self.space_dim)

        self.pos_new = pos.astype(int)

        return self.pos_new

    def evaluate(self, score_new):
        super().evaluate(score_new)

        if score_new < self.score_best:
            self.add_tabu(self.pos_new)

