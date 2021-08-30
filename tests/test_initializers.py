import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(-100, 101, 1),
}


def test_initialize_warm_start_0():
    init = {
        "x1": 0,
    }

    initialize = {"warm_start": [init]}

    opt = RandomSearchOptimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    # print("\nself.results \n", opt.search_data)

    assert abs(opt.best_score) < 0.001


def test_initialize_warm_start_1():
    search_space = {
        "x1": np.arange(-10, 10, 1),
    }
    init = {
        "x1": -10,
    }

    initialize = {"warm_start": [init]}

    opt = RandomSearchOptimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert opt.best_para == init


def test_initialize_warm_start_2():
    search_space = {
        "x1": np.arange(-10, 10, 1),
    }
    init = {
        "x1": -10,
    }

    initialize = {"warm_start": [init], "random": 0, "vertices": 0, "grid": 0}

    opt = RandomSearchOptimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert opt.best_para == init


def test_initialize_vertices():
    initialize = {"vertices": 2}

    opt = RandomSearchOptimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=2)

    assert abs(opt.best_score) - 10000 < 0.001


def test_initialize_grid_0():
    search_space = {
        "x1": np.arange(-1, 2, 1),
    }
    initialize = {"grid": 1}

    opt = RandomSearchOptimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert abs(opt.best_score) < 0.001


def test_initialize_grid_1():
    search_space = {
        "x1": np.arange(-2, 3, 1),
    }

    initialize = {"grid": 1}

    opt = RandomSearchOptimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    assert abs(opt.best_score) - 1 < 0.001
