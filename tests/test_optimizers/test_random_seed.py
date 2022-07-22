import time
import pytest
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(-100, 100, 0.1),
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


@pytest.mark.parametrize(*optimizers)
def test_name_4(Optimizer):
    n_iter = 10

    opt1 = Optimizer(search_space, initialize={"random": 3})
    opt1.search(objective_function, n_iter=n_iter)

    best_score1 = opt1.best_score

    opt2 = Optimizer(
        search_space, initialize={"random": 3}, random_state=opt1.random_seed
    )
    opt2.search(objective_function, n_iter=n_iter)

    best_score2 = opt2.best_score

    assert best_score1 == best_score2
