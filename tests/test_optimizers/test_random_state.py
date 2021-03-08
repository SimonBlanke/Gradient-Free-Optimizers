import pytest
import time
import numpy as np
import pandas as pd


from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100, 1),
}

err = 0.001


@pytest.mark.parametrize(*optimizers)
def test_random_state_0(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": 1})
    opt0.search(
        objective_function,
        n_iter=5,
        random_state=1,
    )

    opt1 = Optimizer(search_space, initialize={"random": 1})
    opt1.search(
        objective_function,
        n_iter=5,
        random_state=1,
    )

    assert abs(opt0.best_score - opt1.best_score) < err


@pytest.mark.parametrize(*optimizers)
def test_random_state_1(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": 1})
    opt0.search(
        objective_function,
        n_iter=5,
        random_state=10,
    )

    opt1 = Optimizer(search_space, initialize={"random": 1})
    opt1.search(
        objective_function,
        n_iter=5,
        random_state=10,
    )

    assert abs(opt0.best_score - opt1.best_score) < err


@pytest.mark.parametrize(*optimizers)
def test_random_state_2(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": 1})
    opt0.search(
        objective_function,
        n_iter=5,
        random_state=1,
    )

    opt1 = Optimizer(search_space, initialize={"random": 1})
    opt1.search(
        objective_function,
        n_iter=5,
        random_state=10,
    )

    assert abs(opt0.best_score - opt1.best_score) > err


@pytest.mark.parametrize(*optimizers)
def test_no_random_state_0(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": 1})
    opt0.search(objective_function, n_iter=5)

    opt1 = Optimizer(search_space, initialize={"random": 1})
    opt1.search(objective_function, n_iter=5)

    assert abs(opt0.best_score - opt1.best_score) > err
