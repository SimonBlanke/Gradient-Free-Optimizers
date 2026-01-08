# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


def _base_para_test_func(opt_para, optimizer):
    opt = optimizer(search_space, initialize={"vertices": 1}, **opt_para)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity=False,
    )

    para_key = list(opt_para.keys())[0]
    para_value = getattr(opt, para_key)

    assert para_value is opt_para[para_key]
