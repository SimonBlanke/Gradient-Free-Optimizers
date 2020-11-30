# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import StochasticHillClimbingOptimizer
from .test_hill_climbing_para_init import hill_climbing_para
from ._base_para_test import _base_para_test_func


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


stochastic_hill_climbing_para = hill_climbing_para + [
    ({"p_accept": 0.01}),
    ({"p_accept": 0.5}),
    ({"p_accept": 1}),
    ({"p_accept": 10}),
    ({"norm_factor": 0.1}),
    ({"norm_factor": 0.5}),
    ({"norm_factor": 0.9}),
    ({"norm_factor": "adaptive"}),
]


pytest_wrapper = ("opt_para", stochastic_hill_climbing_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, StochasticHillClimbingOptimizer)
