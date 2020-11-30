# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest


from gradient_free_optimizers import HillClimbingOptimizer
from ._base_para_test import _base_para_test_func


hill_climbing_para = [
    ({"epsilon": 0.0001}),
    ({"epsilon": 1}),
    ({"epsilon": 10}),
    ({"epsilon": 10000}),
    ({"distribution": "normal"}),
    ({"distribution": "laplace"}),
    ({"distribution": "logistic"}),
    ({"distribution": "gumbel"}),
    ({"n_neighbours": 1}),
    ({"n_neighbours": 10}),
    ({"n_neighbours": 100}),
    ({"rand_rest_p": 0}),
    ({"rand_rest_p": 0.5}),
    ({"rand_rest_p": 1}),
    ({"rand_rest_p": 10}),
]


pytest_wrapper = ("opt_para", hill_climbing_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, HillClimbingOptimizer)
