# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


def _base_para_test_func(opt_para, optimizer):
    opt = optimizer(search_space, **opt_para)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity=False,
        initialize={"vertices": 1},
    )

    para_key = list(opt_para.keys())[0]
    para_value = getattr(opt, para_key)

    assert para_value == opt_para[para_key]
