# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import SimulatedAnnealingOptimizer
from .test_stochastic_hill_climbing_para_init import (
    stochastic_hill_climbing_para,
)
from ._base_para_test import _base_para_test_func


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


simulated_annealing_para = stochastic_hill_climbing_para + [
    ({"annealing_rate": 0.9}),
    ({"annealing_rate": 0.8}),
    ({"annealing_rate": 0.5}),
    ({"annealing_rate": 1}),
    ({"start_temp": 1}),
    ({"start_temp": 0.5}),
    ({"start_temp": 3}),
    ({"start_temp": 10}),
]


pytest_wrapper = ("opt_para", simulated_annealing_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, SimulatedAnnealingOptimizer)
