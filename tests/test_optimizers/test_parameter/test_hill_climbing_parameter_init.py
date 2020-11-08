# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    TabuOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
)


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


HillClimbing_para = [
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


pytest_wrapper = ("para", HillClimbing_para)

optimizers_local = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (TabuOptimizer),
        (SimulatedAnnealingOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
    ],
)


@pytest.mark.parametrize(*optimizers_local)
@pytest.mark.parametrize(*pytest_wrapper)
def test_HillClimbing_para(Optimizer, para):
    opt = Optimizer(search_space, **para)
    opt.search(
        objective_function,
        n_iter=10,
        memory=False,
        verbosity=False,
        initialize={"vertices": 1},
    )

    for optimizer in opt.optimizers:
        para_key = list(para.keys())[0]
        para_value = getattr(optimizer, para_key)

        assert para_value == para[para_key]
