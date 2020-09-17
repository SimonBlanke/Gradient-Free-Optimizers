import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import pytest_parameter


@pytest.mark.parametrize(*pytest_parameter)
def test_exploration_0(Optimizer):
    def objective_function(pos_new):
        score = -(pos_new[0] * pos_new[0] + pos_new[1] * pos_new[1])
        return score

    search_space = [np.arange(-50, 1, 1), np.arange(0, 10, 1)]
    init1 = [-50, 1]
    init2 = [-49, 2]

    opt = Optimizer(search_space)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False,},
        initialize={"warm_start": [init1, init2]},
    )

    print("\n opt.values \n", opt.values, "\n")

    uniques_2nd_dim = list(np.unique(opt.values[:, 1]))

    assert 0 in uniques_2nd_dim


@pytest.mark.parametrize(*pytest_parameter)
def test_exploration_1(Optimizer):
    def objective_function(pos_new):
        score = -(pos_new[0] * pos_new[0] + pos_new[1] * pos_new[1])
        return score

    search_space = [np.arange(-50, 1, 1), np.arange(-10, 1, 1)]
    init1 = [-50, -1]
    init2 = [-49, -2]

    opt = Optimizer(search_space)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False,},
        initialize={"warm_start": [init1]},
    )
    print("\n opt.values \n", opt.values, "\n")

    uniques_2nd_dim = list(np.unique(opt.values[:, 1]))

    assert 0 in uniques_2nd_dim

