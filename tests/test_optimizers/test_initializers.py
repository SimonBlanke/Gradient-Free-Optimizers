import pytest
import numpy as np

from ._parametrize import optimizers, optimizers_2


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(-100, 101, 1),
}


@pytest.mark.parametrize(*optimizers)
def test_initialize_warm_start_0(Optimizer):
    init = {
        "x1": 0,
    }

    initialize = {"warm_start": [init]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    # print("\nself.results \n", opt.search_data)

    assert abs(opt.best_score) < 0.001


@pytest.mark.parametrize(*optimizers)
def test_initialize_warm_start_1(Optimizer):
    search_space = {
        "x1": np.arange(-10, 10, 1),
    }
    init = {
        "x1": -10,
    }

    initialize = {"warm_start": [init]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert opt.best_para == init


@pytest.mark.parametrize(*optimizers)
def test_initialize_warm_start_2(Optimizer):
    search_space = {
        "x1": np.arange(-10, 10, 1),
    }
    init = {
        "x1": -10,
    }

    initialize = {"warm_start": [init], "random": 0, "vertices": 0, "grid": 0}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert opt.best_para == init


@pytest.mark.parametrize(*optimizers)
def test_initialize_vertices(Optimizer):
    initialize = {"vertices": 2}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=2)

    assert abs(opt.best_score) - 10000 < 0.001


@pytest.mark.parametrize(*optimizers)
def test_initialize_grid_0(Optimizer):
    search_space = {
        "x1": np.arange(-1, 2, 1),
    }
    initialize = {"grid": 1}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert abs(opt.best_score) < 0.001


@pytest.mark.parametrize(*optimizers)
def test_initialize_grid_1(Optimizer):
    search_space = {
        "x1": np.arange(-2, 3, 1),
    }

    initialize = {"grid": 1}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert abs(opt.best_score) - 1 < 0.001


@pytest.mark.parametrize(*optimizers)
@pytest.mark.parametrize(*optimizers_2)
def test_initialize_warm_start_twoOpt_0(Optimizer, Optimizer2):
    opt1 = Optimizer(search_space)
    opt1.search(objective_function, n_iter=1)

    opt2 = Optimizer2(search_space, initialize={"warm_start": [opt1.best_para]})
    opt2.search(objective_function, n_iter=20)

    assert opt1.best_score <= opt2.best_score


@pytest.mark.parametrize(*optimizers)
@pytest.mark.parametrize(*optimizers_2)
def test_initialize_warm_start_twoOpt_1(Optimizer, Optimizer2):
    opt1 = Optimizer(search_space)
    opt1.search(objective_function, n_iter=20)

    opt2 = Optimizer2(search_space, initialize={"warm_start": [opt1.best_para]})
    opt2.search(objective_function, n_iter=1)

    assert opt1.best_score <= opt2.best_score
