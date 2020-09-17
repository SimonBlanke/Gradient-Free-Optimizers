# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import StochasticHillClimbingOptimizer
from ._base_test import _base_test

n_iter = 33
opt = StochasticHillClimbingOptimizer


def test_epsilon():
    for epsilon in [0.00001, 100]:
        opt_para = {"epsilon": epsilon}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_n_neighbours():
    for n_neighbours in [1, 100]:
        opt_para = {"n_neighbours": n_neighbours}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_p_accept():
    for p_accept in [0.0001, 100]:
        opt_para = {"p_accept": p_accept}
        _base_test(opt, n_iter, opt_para=opt_para)
