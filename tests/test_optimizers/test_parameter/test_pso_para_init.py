# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import ParticleSwarmOptimizer
from ._base_para_test import _base_para_test_func


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


parallel_tempering_para = [
    ({"inertia": 0.5}),
    ({"inertia": 0.9}),
    ({"inertia": 0.1}),
    ({"inertia": 0}),
    ({"inertia": 2}),
    ({"cognitive_weight": 0.5}),
    ({"cognitive_weight": 0.9}),
    ({"cognitive_weight": 0.1}),
    ({"cognitive_weight": 0}),
    ({"cognitive_weight": 2}),
    ({"social_weight": 0.5}),
    ({"social_weight": 0.9}),
    ({"social_weight": 0.1}),
    ({"social_weight": 0}),
    ({"social_weight": 2}),
    ({"temp_weight": 0.5}),
    ({"temp_weight": 0.9}),
    ({"temp_weight": 0.1}),
    ({"temp_weight": 0}),
    ({"temp_weight": 2}),
    ({"population": 1}),
    ({"population": 2}),
    ({"population": 100}),
    ({"rand_rest_p": 0}),
    ({"rand_rest_p": 0.5}),
    ({"rand_rest_p": 1}),
]


pytest_wrapper = ("opt_para", parallel_tempering_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, ParticleSwarmOptimizer)
