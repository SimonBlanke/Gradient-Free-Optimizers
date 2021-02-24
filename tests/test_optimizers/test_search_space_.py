import pytest
import random
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    return -(para["x1"] + para["x1"])


search_space1 = {
    "x1": np.array([1]),
    "x2": np.arange(-10, 10, 1),
}

search_space2 = {
    "x1": np.arange(-10, 10, 1),
    "x2": np.array([1]),
}


objective_para = (
    "search_space",
    [
        (search_space1),
        (search_space2),
    ],
)


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*optimizers)
def test_best_results_0(Optimizer, search_space):
    opt = Optimizer(search_space)
    opt.search(objective_function, n_iter=30)
