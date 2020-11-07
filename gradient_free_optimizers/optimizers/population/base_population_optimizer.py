# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from ...converter import Converter


class BasePopulationOptimizer:
    def __init__(self, search_space):
        super().__init__()
        conv = Converter(search_space)
        self.conv = conv

        self.eval_times = []
        self.iter_times = []

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter
