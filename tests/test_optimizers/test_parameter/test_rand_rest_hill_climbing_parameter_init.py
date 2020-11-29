# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import RandomRestartHillClimbingOptimizer
from .test_hill_climbing_parameter_init import hill_climbing_para


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


rand_rest_hill_climbing_para = hill_climbing_para + [
    ({"n_iter_restart": 1}),
    ({"n_iter_restart": 10}),
    ({"n_iter_restart": 100}),
]


pytest_wrapper = ("opt_para", rand_rest_hill_climbing_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_rand_rest_hill_climbing_para(opt_para):
    opt = RandomRestartHillClimbingOptimizer(search_space, **opt_para)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity=False,
        initialize={"vertices": 1},
    )

    for optimizer in opt.optimizers:
        para_key = list(opt_para.keys())[0]
        para_value = getattr(optimizer, para_key)

        assert para_value == opt_para[para_key]
