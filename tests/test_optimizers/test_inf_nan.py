import pytest
import random
import numpy as np

from ._parametrize import optimizers


def objective_function_nan(para):
    rand = random.randint(0, 1)

    if rand == 0:
        return 1
    else:
        return np.nan


def objective_function_m_inf(para):
    rand = random.randint(0, 1)

    if rand == 0:
        return 1
    else:
        return -np.inf


def objective_function_inf(para):
    rand = random.randint(0, 1)

    if rand == 0:
        return 1
    else:
        return np.inf


search_space = {"x1": np.arange(0, 20, 1)}


objective_para = (
    "objective",
    [
        (objective_function_nan),
        (objective_function_m_inf),
        (objective_function_inf),
    ],
)


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*optimizers)
def test_inf_nan_0(Optimizer, objective):
    objective_function = objective
    initialize = {"random": 20}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(
        objective_function,
        n_iter=80,
        verbosity={"print_results": False, "progress_bar": False},
    )


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*optimizers)
def test_inf_nan_1(Optimizer, objective):
    objective_function = objective
    initialize = {"random": 20}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False},
    )

    search_data = opt.search_data
    print("\n search_data \n", search_data)

    non_inf_mask = ~np.isinf(search_data["score"].values)
    non_nan_mask = ~np.isnan(search_data["score"].values)

    non_inf_nan = np.sum(non_inf_mask * non_nan_mask)

    assert 10 < non_inf_nan < 40
