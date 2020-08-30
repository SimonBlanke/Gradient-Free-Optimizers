# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import StochasticHillClimbingOptimizer
from ._base_test import _base_test

n_iter = 33
opt = StochasticHillClimbingOptimizer


def test_p_accept():
    for p_accept in [0.0001, 100]:
        opt_para = {"p_accept": p_accept}
        _base_test(opt, n_iter, opt_para=opt_para)
