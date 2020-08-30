# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


from . import HillClimbingOptimizer
from ...search import Search
from scipy.spatial.distance import euclidean


def gaussian(distance, sig, sigma_factor=1):
    return (
        sigma_factor
        * sig
        * np.exp(-np.power(distance, 2.0) / (sigma_factor * np.power(sig, 2.0)))
    )


class TabuOptimizer(HillClimbingOptimizer, Search):
    def __init__(self, search_space, tabu_factor=3):
        super().__init__(search_space)

        self.tabus = []
        self.tabu_factor = tabu_factor
        self.epsilon_mod = 1

    def iterate(self):
        return self._move_climb(self.pos_current)

    def evaluate(self, score_new):
        super().evaluate(score_new)

        if score_new <= self.score_current:
            self.epsilon_mod = self.tabu_factor
        else:
            self.epsilon_mod = 1
