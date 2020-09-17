# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import TabuOptimizer
from ._base_test import _base_test

n_iter = 33
opt = TabuOptimizer


def test_epsilon():
    for epsilon in [0.00001, 100]:
        opt_para = {"epsilon": epsilon}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_n_neighbours():
    for n_neighbours in [1, 100]:
        opt_para = {"n_neighbours": n_neighbours}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_tabu_factor():
    for tabu_factor in [1, 3, 20, 50, 100]:
        opt_para = {"tabu_factor": tabu_factor}
        _base_test(opt, n_iter, opt_para=opt_para)
