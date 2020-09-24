import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import pytest_parameter


@pytest.mark.parametrize(*pytest_parameter)
def test_exploration_0(Optimizer):
    def objective_function(para):
        score = -(para["x1"] * para["x1"] + para["x2"] * para["x2"])
        return score

    search_space = {
        "x1": np.arange(-50, 1, 1),
        "x2": np.arange(0, 10, 1),
    }

    init1 = {
        "x1": -50,
        "x2": 1,
    }

    init2 = {
        "x1": -49,
        "x2": 2,
    }

    opt = Optimizer(search_space)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False,},
        initialize={"warm_start": [init1, init2]},
    )

    uniques_2nd_dim = list(opt.results["x2"].values)

    print("\n uniques_2nd_dim \n", uniques_2nd_dim, "\n")
    print("\n Results head \n", opt.results.head())
    print("\n Results tail \n", opt.results.tail())

    print("\nN iter:", len(opt.results))

    assert 0 in uniques_2nd_dim


@pytest.mark.parametrize(*pytest_parameter)
def test_exploration_1(Optimizer):
    def objective_function(para):
        score = -(para["x1"] * para["x1"] + para["x2"] * para["x2"])
        return score

    search_space = {
        "x1": np.arange(-50, 1, 1),
        "x2": np.arange(-10, 1, 1),
    }

    init1 = {
        "x1": -50,
        "x2": -1,
    }

    init2 = {
        "x1": -49,
        "x2": -2,
    }

    opt = Optimizer(search_space)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False,},
        initialize={"warm_start": [init1]},
    )

    uniques_2nd_dim = list(opt.results["x2"].values)

    print("\n uniques_2nd_dim \n", uniques_2nd_dim, "\n")
    print("\n Results head \n", opt.results.head())
    print("\n Results tail \n", opt.results.tail())

    print("\nN iter:", len(opt.results))

    assert 0 in uniques_2nd_dim

