import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from gradient_free_optimizers import RandomSearchOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100000, 0.1),
}


def test_attributes_results_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert isinstance(opt.results, pd.DataFrame)


def test_attributes_results_1():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert set(search_space.keys()) < set(opt.results.columns)


def test_attributes_results_2():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert "x1" in list(opt.results.columns)


def test_attributes_results_3():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert "score" in list(opt.results.columns)


def test_attributes_best_score_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert np.inf > opt.best_score > -np.inf


def test_attributes_best_value_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert isinstance(opt.best_value, list)
