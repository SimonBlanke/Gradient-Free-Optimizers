import pytest
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 1, 1)}


@pytest.mark.parametrize(*optimizers)
def test_searches_0(Optimizer):

    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=1)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 2


@pytest.mark.parametrize(*optimizers)
def test_searches_1(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=10)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 11


@pytest.mark.parametrize(*optimizers)
def test_searches_2(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=20)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 21


@pytest.mark.parametrize(*optimizers)
def test_searches_3(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=20)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 30


@pytest.mark.parametrize(*optimizers)
def test_searches_4(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=10)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 30
