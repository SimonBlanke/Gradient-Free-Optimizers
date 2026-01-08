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


def test_attributes_best_score_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert np.inf > opt.best_score > -np.inf


def test_attributes_best_para_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert isinstance(opt.best_para, dict)


def test_attributes_best_para_1():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert list(opt.best_para.keys()) == list(search_space.keys())


def test_attributes_eval_times_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert isinstance(opt.eval_times, list)


def test_attributes_eval_times_1():
    c_time = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)
    diff_time = time.time() - c_time

    assert np.array(opt.eval_times).sum() < diff_time


def test_attributes_iter_times_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert isinstance(opt.iter_times, list)


def test_attributes_iter_times_1():
    c_time = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)
    diff_time = time.time() - c_time

    assert np.array(opt.iter_times).sum() < diff_time
