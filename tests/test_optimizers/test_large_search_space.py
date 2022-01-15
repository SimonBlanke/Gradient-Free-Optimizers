import pytest
import numpy as np

from ._parametrize import optimizers_noSBOM


def objective_function(para):
    return 1


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_large_search_space_0(Optimizer):

    search_space = {
        "x1": np.arange(0, 1000000),
        "x2": np.arange(0, 1000000),
        "x3": np.arange(0, 1000000),
    }
    opt = Optimizer(search_space, initialize={"random": 10})
    opt.search(objective_function, n_iter=15, verbosity=False)


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_large_search_space_1(Optimizer):

    search_space = {
        "x1": np.arange(0, 1000, 0.001),
        "x2": np.arange(0, 1000, 0.001),
        "x3": np.arange(0, 1000, 0.001),
    }

    opt = Optimizer(search_space, initialize={"random": 10})
    opt.search(objective_function, n_iter=15, verbosity=False)


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_large_search_space_2(Optimizer):

    search_space = {}
    for i in range(100):
        key = "x" + str(i)
        search_space[key] = np.arange(0, 3)

    opt = Optimizer(search_space, initialize={"random": 3, "vertices": 3, "grid": 3})
    opt.search(objective_function, n_iter=10, verbosity=False)
