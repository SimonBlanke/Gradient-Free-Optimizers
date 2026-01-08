# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import RepulsingHillClimbingOptimizer
from .test_hill_climbing_para_init import hill_climbing_para
from ._base_para_test import _base_para_test_func


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


tabu_search = hill_climbing_para + [
    ({"repulsion_factor": 1}),
    ({"repulsion_factor": 2}),
    ({"repulsion_factor": 2.5}),
    ({"repulsion_factor": 10}),
]


pytest_wrapper = ("opt_para", tabu_search)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, RepulsingHillClimbingOptimizer)
