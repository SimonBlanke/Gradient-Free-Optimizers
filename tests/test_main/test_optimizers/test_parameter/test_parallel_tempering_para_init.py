# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import ParallelTemperingOptimizer
from ._base_para_test import _base_para_test_func
from gradient_free_optimizers import SimulatedAnnealingOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


parallel_tempering_para = [
    ({"n_iter_swap": 1}),
    ({"n_iter_swap": 2}),
    ({"n_iter_swap": 10}),
    ({"n_iter_swap": 100}),
    ({"population": 1}),
    ({"population": 2}),
    ({"population": 100}),
    (
        {
            "population": [
                SimulatedAnnealingOptimizer(search_space),
                SimulatedAnnealingOptimizer(search_space),
                SimulatedAnnealingOptimizer(search_space),
                SimulatedAnnealingOptimizer(search_space),
            ]
        }
    ),
    ({"rand_rest_p": 0}),
    ({"rand_rest_p": 0.5}),
    ({"rand_rest_p": 1}),
]


pytest_wrapper = ("opt_para", parallel_tempering_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, ParallelTemperingOptimizer)
