import time
import pytest
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100, 1),
}


@pytest.mark.parametrize(*optimizers)
def test_name_0(Optimizer):
    opt = Optimizer(search_space, initialize={"random": 3})
    opt.search(objective_function, n_iter=5)

    assert opt.random_seed is not None


@pytest.mark.parametrize(*optimizers)
def test_name_1(Optimizer):
    opt = Optimizer(search_space)

    assert opt.random_seed is not None


@pytest.mark.parametrize(*optimizers)
def test_name_2(Optimizer):
    random_state = 42
    opt = Optimizer(search_space, initialize={"random": 3}, random_state=random_state)
    opt.search(objective_function, n_iter=5)

    assert opt.random_seed == random_state


@pytest.mark.parametrize(*optimizers)
def test_name_3(Optimizer):
    random_state = 42
    opt = Optimizer(search_space, random_state=random_state)

    assert opt.random_seed == random_state
