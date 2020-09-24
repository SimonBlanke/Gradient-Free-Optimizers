import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from gradient_free_optimizers import RandomSearchOptimizer, HillClimbingOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100000, 0.1),
}

err = 0.01


def test_random_state_0():
    opt0 = RandomSearchOptimizer(search_space)
    opt0.search(
        objective_function, n_iter=100, initialize={"random": 1}, random_state=1
    )

    opt1 = RandomSearchOptimizer(search_space)
    opt1.search(
        objective_function, n_iter=100, initialize={"random": 1}, random_state=1
    )

    assert abs(opt0.best_score - opt1.best_score) < err


def test_random_state_1():
    opt0 = RandomSearchOptimizer(search_space)
    opt0.search(
        objective_function, n_iter=100, initialize={"random": 1}, random_state=10
    )

    opt1 = RandomSearchOptimizer(search_space)
    opt1.search(
        objective_function, n_iter=100, initialize={"random": 1}, random_state=10
    )

    assert abs(opt0.best_score - opt1.best_score) < err


def test_random_state_2():
    opt0 = RandomSearchOptimizer(search_space)
    opt0.search(
        objective_function, n_iter=100, initialize={"random": 1}, random_state=1
    )

    opt1 = RandomSearchOptimizer(search_space)
    opt1.search(
        objective_function, n_iter=100, initialize={"random": 1}, random_state=10
    )

    assert abs(opt0.best_score - opt1.best_score) > err


def test_no_random_state_0():
    opt0 = RandomSearchOptimizer(search_space)
    opt0.search(objective_function, n_iter=100, initialize={"random": 1})

    opt1 = RandomSearchOptimizer(search_space)
    opt1.search(objective_function, n_iter=100, initialize={"random": 1})

    assert abs(opt0.best_score - opt1.best_score) > err
