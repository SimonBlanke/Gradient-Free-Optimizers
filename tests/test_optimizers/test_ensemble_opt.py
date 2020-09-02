# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import EnsembleOptimizer
from ._base_test import _base_test

n_iter = 33
opt = EnsembleOptimizer


def test_opt():
    _base_test(opt, n_iter)

