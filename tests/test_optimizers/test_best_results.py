import pytest
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


def objective_function_m5(para):
    score = -(para["x1"] - 5) * (para["x1"] - 5)
    return score


def objective_function_p5(para):
    score = -(para["x1"] + 5) * (para["x1"] + 5)
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


objective_para = (
    "objective",
    [
        (objective_function, search_space),
        (objective_function_m5, search_space),
        (objective_function_p5, search_space),
    ],
)


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*optimizers)
def test_best_results_0(Optimizer, objective):
    search_space = objective[1]
    objective_function = objective[0]

    initialize = {"vertices": 2}

    opt = Optimizer(search_space)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
        initialize=initialize,
    )

    assert opt.best_score == objective_function(opt.best_para)


@pytest.mark.parametrize(*optimizers)
def test_best_results_1(Optimizer):
    search_space = {"x1": np.arange(-100, 101, 1)}

    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    initialize = {"vertices": 2}

    opt = Optimizer(search_space)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
        initialize=initialize,
    )

    assert opt.best_para["x1"] in list(opt.results["x1"])

