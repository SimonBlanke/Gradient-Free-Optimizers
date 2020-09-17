# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import RandomAnnealingOptimizer
from ._base_test import _base_test

n_iter = 33
opt = RandomAnnealingOptimizer


def test_epsilon():
    for epsilon in [0.00001, 100]:
        opt_para = {"epsilon": epsilon}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_n_neighbours():
    for n_neighbours in [1, 100]:
        opt_para = {"n_neighbours": n_neighbours}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_annealing_rate():
    for annealing_rate in [1, 0.001]:
        opt_para = {"annealing_rate": annealing_rate}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_start_temp():
    for start_temp in [0.001, 10000]:
        opt_para = {"start_temp": start_temp}
        _base_test(opt, n_iter, opt_para=opt_para)
