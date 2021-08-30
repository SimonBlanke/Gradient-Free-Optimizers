import pytest
import numpy as np

from ._parametrize import optimizers


@pytest.mark.parametrize(*optimizers)
def test_results_0(Optimizer):
    search_space = {"x1": np.arange(-10, 1, 1)}

    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    initialize = {"random": 2}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
    )

    results_set = set(opt.search_data["x1"])
    search_space_set = set(search_space["x1"])

    assert results_set.issubset(search_space_set)


@pytest.mark.parametrize(*optimizers)
def test_results_1(Optimizer):
    search_space = {"x1": np.arange(-10, 1, 1), "x2": np.arange(-10, 1, 1)}

    def objective_function(para):
        score = -(para["x1"] * para["x1"] + para["x2"] * para["x2"])
        return score

    initialize = {"random": 2}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
    )

    results_set_x1 = set(opt.search_data["x1"])
    search_space_set_x1 = set(search_space["x1"])

    assert results_set_x1.issubset(search_space_set_x1)

    results_set_x2 = set(opt.search_data["x2"])
    search_space_set_x2 = set(search_space["x2"])

    assert results_set_x2.issubset(search_space_set_x2)
