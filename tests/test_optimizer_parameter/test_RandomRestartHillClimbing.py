# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import RandomRestartHillClimbingOptimizer
from ._base_test import _base_test

n_iter = 33
opt = RandomRestartHillClimbingOptimizer


def test_n_iter_restart():
    for n_iter_restart in [1, 50, 100]:
        opt_para = {"n_iter_restart": n_iter_restart}
        _base_test(opt, n_iter, opt_para=opt_para)
