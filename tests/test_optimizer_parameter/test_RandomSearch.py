# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import RandomSearchOptimizer
from ._base_test import _base_test

n_iter = 33
opt = RandomSearchOptimizer


def test_():
    _base_test(opt, n_iter)
