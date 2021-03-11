# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import EvolutionStrategyOptimizer
from ._base_para_test import _base_para_test_func
from gradient_free_optimizers import HillClimbingOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


parallel_tempering_para = [
    ({"mutation_rate": 0.5}),
    ({"mutation_rate": 0.9}),
    ({"mutation_rate": 0.1}),
    ({"mutation_rate": 0}),
    ({"mutation_rate": 2}),
    ({"crossover_rate": 0.5}),
    ({"crossover_rate": 0.9}),
    ({"crossover_rate": 0.1}),
    ({"crossover_rate": 0}),
    ({"crossover_rate": 2}),
    ({"population": 1}),
    ({"population": 2}),
    ({"population": 100}),
    (
        {
            "population": [
                HillClimbingOptimizer(search_space),
                HillClimbingOptimizer(search_space),
                HillClimbingOptimizer(search_space),
                HillClimbingOptimizer(search_space),
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
    _base_para_test_func(opt_para, EvolutionStrategyOptimizer)
