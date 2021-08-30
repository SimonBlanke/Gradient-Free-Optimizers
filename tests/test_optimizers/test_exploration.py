import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import optimizers_singleOpt


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_exploration_0(Optimizer):
    def objective_function(para):
        score = -(para["x1"] * para["x1"])
        return score

    search_space = {
        "x1": np.arange(0, 11, 1),
    }

    init1 = {
        "x1": 1,
    }

    init2 = {
        "x1": 9,
    }
    opt = Optimizer(search_space, initialize={"warm_start": [init1, init2]})
    opt.search(
        objective_function,
        n_iter=300,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
    )

    uniques_2nd_dim = list(opt.search_data["x1"].values)

    print("\n uniques_2nd_dim \n", uniques_2nd_dim, "\n")
    print("\n Results head \n", opt.search_data.head())
    print("\n Results tail \n", opt.search_data.tail())

    print("\nN iter:", len(opt.search_data))

    assert 0 in uniques_2nd_dim


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_exploration_1(Optimizer):
    def objective_function(para):
        score = -(para["x1"] * para["x1"])
        return score

    search_space = {
        "x1": np.arange(-10, 1, 1),
    }

    init1 = {
        "x1": -1,
    }

    init2 = {
        "x1": -9,
    }

    opt = Optimizer(search_space, initialize={"warm_start": [init1, init2]})
    opt.search(
        objective_function,
        n_iter=300,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
    )

    uniques_2nd_dim = list(opt.search_data["x1"].values)

    print("\n uniques_2nd_dim \n", uniques_2nd_dim, "\n")
    print("\n Results head \n", opt.search_data.head())
    print("\n Results tail \n", opt.search_data.tail())

    print("\nN iter:", len(opt.search_data))

    assert 0 in uniques_2nd_dim
