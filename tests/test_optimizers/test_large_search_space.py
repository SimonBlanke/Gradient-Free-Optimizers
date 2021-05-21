import pytest
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    return 1


@pytest.mark.parametrize(*optimizers)
def test_large_search_space_0(Optimizer):

    search_space = {
        "x1": np.arange(0, 1000000),
        "x2": np.arange(0, 1000000),
        "x3": np.arange(0, 1000000),
    }
    opt = Optimizer(search_space, initialize={"random": 3})
    opt.search(objective_function, n_iter=5, verbosity=False)


@pytest.mark.parametrize(*optimizers)
def test_large_search_space_1(Optimizer):

    search_space = {
        "x1": np.arange(0, 1000, 0.001),
        "x2": np.arange(0, 1000, 0.001),
        "x3": np.arange(0, 1000, 0.001),
    }

    opt = Optimizer(search_space, initialize={"random": 3})
    opt.search(objective_function, n_iter=5, verbosity=False)


"""
@pytest.mark.parametrize(*optimizers)
def test_large_search_space_2(Optimizer):

    search_space = {
        "x1": np.arange(0, 3),
        "x2": np.arange(0, 3),
        "x3": np.arange(0, 3),
        "x4": np.arange(0, 3),
        "x5": np.arange(0, 3),
        "x6": np.arange(0, 3),
        "x7": np.arange(0, 3),
        "x8": np.arange(0, 3),
        "x9": np.arange(0, 3),
        "x10": np.arange(0, 3),
        "x11": np.arange(0, 3),
        "x12": np.arange(0, 3),
        "x13": np.arange(0, 3),
        "x14": np.arange(0, 3),
        "x15": np.arange(0, 3),
        "x16": np.arange(0, 3),
        "x17": np.arange(0, 3),
        "x18": np.arange(0, 3),
        "x19": np.arange(0, 3),
        "x20": np.arange(0, 3),
        "x21": np.arange(0, 3),
        "x22": np.arange(0, 3),
        "x23": np.arange(0, 3),
        "x24": np.arange(0, 3),
        "x25": np.arange(0, 3),
        "x26": np.arange(0, 3),
        "x27": np.arange(0, 3),
        "x28": np.arange(0, 3),
        "x29": np.arange(0, 3),
        "x30": np.arange(0, 3),
        "x31": np.arange(0, 3),
        "x32": np.arange(0, 3),
        "x33": np.arange(0, 3),
    }

    opt = Optimizer(search_space, initialize={"random": 3})
    opt.search(objective_function, n_iter=5, verbosity=False)
"""
