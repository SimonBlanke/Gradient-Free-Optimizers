# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class DownhillSimplexOptimizer(BaseOptimizer):
    def __init__(self, _opt_args_):
        pass

    def _iterate(self, i, _cand_):
        pass

    def _init_iteration(self, _cand_):
        pass


class DownhillSimplexPositioner(BasePositioner):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)

        self.epsilon = _opt_args_.epsilon
        self.distribution = _opt_args_.distribution
