import pytest
import time
import numpy as np
import pandas as pd


from ._parametrize import optimizers
from surfaces.test_functions import AckleyFunction


ackkley_function = AckleyFunction()


def objective_function(para):
    score = -(para["x0"] * para["x0"] + para["x1"] * para["x1"])
    return score


search_space = {
    "x0": np.arange(-75, 100, 1),
    "x1": np.arange(-100, 75, 1),
}

err = 0.000001

n_iter = 10
n_random = 2

n_last = n_iter - n_random


@pytest.mark.parametrize(*optimizers)
def test_random_state_0(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": n_random}, random_state=1)
    opt0.search(
        ackkley_function,
        n_iter=n_iter,
    )

    opt1 = Optimizer(search_space, initialize={"random": n_random}, random_state=1)
    opt1.search(
        ackkley_function,
        n_iter=n_iter,
    )

    print("\n opt0.search_data \n", opt0.search_data)
    print("\n opt1.search_data \n", opt1.search_data)

    n_last_scores0 = list(opt0.search_data["score"].values)[-n_last:]
    n_last_scores1 = list(opt1.search_data["score"].values)[-n_last:]

    assert abs(np.sum(n_last_scores0) - np.sum(n_last_scores1)) < err


@pytest.mark.parametrize(*optimizers)
def test_random_state_1(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": n_random}, random_state=10)
    opt0.search(
        ackkley_function,
        n_iter=n_iter,
    )

    opt1 = Optimizer(search_space, initialize={"random": n_random}, random_state=10)
    opt1.search(
        ackkley_function,
        n_iter=n_iter,
    )

    n_last_scores0 = list(opt0.search_data["score"].values)[-n_last:]
    n_last_scores1 = list(opt1.search_data["score"].values)[-n_last:]

    assert abs(np.sum(n_last_scores0) - np.sum(n_last_scores1)) < err


@pytest.mark.parametrize(*optimizers)
def test_random_state_2(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": n_random}, random_state=1)
    opt0.search(
        ackkley_function,
        n_iter=n_iter,
    )

    opt1 = Optimizer(search_space, initialize={"random": n_random}, random_state=10)
    opt1.search(
        ackkley_function,
        n_iter=n_iter,
    )

    print("\n opt0.search_data \n", opt0.search_data)
    print("\n opt1.search_data \n", opt1.search_data)

    n_last_scores0 = list(opt0.search_data["score"].values)[-n_last:]
    n_last_scores1 = list(opt1.search_data["score"].values)[-n_last:]

    assert abs(np.sum(n_last_scores0) - np.sum(n_last_scores1)) > err


@pytest.mark.parametrize(*optimizers)
def test_no_random_state_0(Optimizer):
    opt0 = Optimizer(search_space, initialize={"random": n_random})
    opt0.search(ackkley_function, n_iter=n_iter)

    opt1 = Optimizer(search_space, initialize={"random": n_random})
    opt1.search(ackkley_function, n_iter=n_iter)

    print("\n opt0.search_data \n", opt0.search_data)
    print("\n opt1.search_data \n", opt1.search_data)

    n_last_scores0 = list(opt0.search_data["score"].values)[-n_last:]
    n_last_scores1 = list(opt1.search_data["score"].values)[-n_last:]

    assert abs(np.sum(n_last_scores0) - np.sum(n_last_scores1)) > err
