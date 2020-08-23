# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import DecisionTreeOptimizer
from ._base_test import _base_test

n_iter = 33
opt = DecisionTreeOptimizer


"""
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
"""


def test_max_sample_size():
    for max_sample_size in [10, 100, 10000, 10000000000]:
        opt_para = {"max_sample_size": max_sample_size}
        _base_test(opt, n_iter, opt_para=opt_para)


"""
def test_gpr():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(
        search_config,
        n_iter=n_iter,
        optimizer={"DecisionTree": {"tree_regressor": "random_forest"}},
    )
"""
