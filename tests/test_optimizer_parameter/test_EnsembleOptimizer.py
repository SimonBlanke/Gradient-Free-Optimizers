# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import EnsembleOptimizer
from ._base_test import _base_test

n_iter = 33
opt = EnsembleOptimizer


def get_score(pos_new):
    return -(pos_new[0] * pos_new[0])


def test_warm_start_smbo():
    gpr_X, gpr_y = [], []
    for _ in range(10):
        pos = np.random.randint(0, high=9)
        pos = np.array([pos])
        gpr_X.append(pos)
        gpr_y.append(get_score(pos))

    for warm_start_smbo in [None, (gpr_X, gpr_y)]:
        opt_para = {"warm_start_smbo": warm_start_smbo}
        _base_test(opt, n_iter, opt_para=opt_para)
