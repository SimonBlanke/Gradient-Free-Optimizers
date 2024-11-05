import pytest
import numpy as np

from ._parametrize import optimizers_noSBOM, optimizers_SBOM


def objective_function(para):
    return 1


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_large_search_space_0(Optimizer):

    search_space = {
        "x1": np.arange(0, 100000),
        "x2": np.arange(0, 100000),
        "x3": np.arange(0, 100000),
    }
    opt = Optimizer(search_space, initialize={"random": 10})
    opt.search(objective_function, n_iter=150, verbosity=False)


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_large_search_space_1(Optimizer):

    search_space = {
        "x1": np.arange(0, 100, 0.001),
        "x2": np.arange(0, 100, 0.001),
        "x3": np.arange(0, 100, 0.001),
    }

    opt = Optimizer(search_space, initialize={"random": 10})
    opt.search(objective_function, n_iter=150, verbosity=False)


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_large_search_space_2(Optimizer):

    search_space = {}
    for i in range(33):
        key = "x" + str(i)
        search_space[key] = np.arange(0, 100)

    opt = Optimizer(search_space, initialize={"random": 34, "vertices": 34, "grid": 34})
    opt.search(objective_function, n_iter=1000, verbosity=False)


@pytest.mark.parametrize(*optimizers_SBOM)
def test_large_search_space_3(Optimizer):

    search_space = {}
    for i in range(10):
        key = "x" + str(i)
        search_space[key] = np.arange(0, 10)

    opt = Optimizer(search_space, initialize={"random": 1})
    opt.search(objective_function, n_iter=2, verbosity=False)
