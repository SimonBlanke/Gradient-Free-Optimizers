# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest


from gradient_free_optimizers import GridSearchOptimizer
from ._base_para_test import _base_para_test_func


grid_search_para = [
    ({"step_size": 1}),
    ({"step_size": 10}),
    ({"step_size": 10000}),
    ({"direction": "diagonal"}),
    ({"direction": "orthogonal"}),
]


pytest_wrapper = ("opt_para", grid_search_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_grid_search_para(opt_para):
    _base_para_test_func(opt_para, GridSearchOptimizer)
