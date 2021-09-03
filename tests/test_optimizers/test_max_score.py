import pytest
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100000, 0.1),
}


@pytest.mark.parametrize(*optimizers)
def test_max_score_0(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 100, 0.1),
    }

    max_score = -9999

    opt = Optimizer(
        search_space,
        initialize={"warm_start": [{"x1": 99}]},
    )
    opt.search(
        objective_function,
        n_iter=100,
        max_score=max_score,
    )

    print("\n Results head \n", opt.search_data.head())
    print("\n Results tail \n", opt.search_data.tail())

    print("\nN iter:", len(opt.search_data))

    assert -100 > opt.best_score > max_score


@pytest.mark.parametrize(*optimizers)
def test_max_score_1(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        time.sleep(0.01)
        return score

    search_space = {
        "x1": np.arange(0, 100, 0.1),
    }

    max_score = -9999

    c_time = time.time()
    opt = Optimizer(search_space, initialize={"warm_start": [{"x1": 99}]})
    opt.search(
        objective_function,
        n_iter=100000,
        max_score=max_score,
    )
    diff_time = time.time() - c_time

    print("\n Results head \n", opt.search_data.head())
    print("\n Results tail \n", opt.search_data.tail())

    print("\nN iter:", len(opt.search_data))

    assert diff_time < 1
