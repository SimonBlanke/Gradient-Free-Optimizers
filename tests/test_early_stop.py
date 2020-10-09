import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from gradient_free_optimizers import (
    RandomSearchOptimizer,
    HillClimbingOptimizer,
)


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100000, 0.1),
}


def test_max_time_0():
    c_time1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=1000000, max_time=0.1)
    diff_time1 = time.time() - c_time1

    assert diff_time1 < 1


def test_max_time_1():
    c_time1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=1000000, max_time=1)
    diff_time1 = time.time() - c_time1

    assert 0.3 < diff_time1 < 2


def test_max_score_0():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 100, 0.1),
    }

    max_score = -9999

    opt = HillClimbingOptimizer(search_space, epsilon=0.01, rand_rest_p=0)
    opt.search(
        objective_function,
        n_iter=100000,
        initialize={},
        warm_start=[{"x1": 99}],
        max_score=max_score,
    )

    print("\n Results head \n", opt.results.head())
    print("\n Results tail \n", opt.results.tail())

    print("\nN iter:", len(opt.results))

    assert -100 > opt.best_score > max_score


def test_max_score_1():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        time.sleep(0.01)
        return score

    search_space = {
        "x1": np.arange(0, 100, 0.1),
    }

    max_score = -9999

    c_time = time.time()
    opt = HillClimbingOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=100000,
        initialize={},
        warm_start=[{"x1": 99}],
        max_score=max_score,
    )
    diff_time = time.time() - c_time

    print("\n Results head \n", opt.results.head())
    print("\n Results tail \n", opt.results.tail())

    print("\nN iter:", len(opt.results))

    assert diff_time < 1


test_max_score_1()
