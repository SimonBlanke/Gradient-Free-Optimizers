# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from ...converter import Converter
from ...results_manager import ResultsManager


class BasePopulationOptimizer:
    def __init__(
        self, search_space, initialize={"grid": 4, "random": 2, "vertices": 4},
    ):
        super().__init__()
        self.conv = Converter(search_space)
        self.results_mang = ResultsManager(self.conv)
        self.initialize = initialize

        self.eval_times = []
        self.iter_times = []

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter
