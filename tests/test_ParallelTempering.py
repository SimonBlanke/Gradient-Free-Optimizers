# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import ParallelTemperingOptimizer
from ._base_test import _base_test

n_iter = 100
opt = ParallelTemperingOptimizer


def test_n_swaps():
    for n_iter_swap in [1, 3, 10, 33]:
        opt_para = {"n_iter_swap": n_iter_swap}
        _base_test(opt, n_iter, opt_para=opt_para)
