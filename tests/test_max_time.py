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
