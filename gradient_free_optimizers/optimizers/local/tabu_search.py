# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


from . import HillClimbingOptimizer
from ...search import Search
from scipy.spatial.distance import euclidean


class TabuOptimizer(HillClimbingOptimizer, Search):
    def __init__(self, search_space, tabu_factor=3, **kwargs):
        super().__init__(search_space)

        self.tabus = []
        self.tabu_factor = tabu_factor
        self.epsilon_mod = 1

    @HillClimbingOptimizer.iter_dec
    def iterate(self):
        return self._move_climb(self.pos_current)

    def evaluate(self, score_new):
        super().evaluate(score_new)

        if score_new <= self.score_current:
            self.epsilon_mod = self.tabu_factor
        else:
            self.epsilon_mod = 1

