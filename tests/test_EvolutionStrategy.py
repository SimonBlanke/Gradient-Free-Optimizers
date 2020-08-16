# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import EvolutionStrategyOptimizer
from ._base_test import _base_test

n_iter = 100
opt = EvolutionStrategyOptimizer


def test_mutation_rate():
    for mutation_rate in [0.1, 0.9]:
        opt_para = {"mutation_rate": mutation_rate}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_crossover_rate():
    for crossover_rate in [0.1, 0.9]:
        opt_para = {"crossover_rate": crossover_rate}
        _base_test(opt, n_iter, opt_para=opt_para)
