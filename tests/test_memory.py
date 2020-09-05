import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from gradient_free_optimizers import RandomSearchOptimizer


def objective_function(values):
    score = -values[0] * values[0]
    return score


search_space = [np.arange(-1, 1, 1)]


def test_memory_time_save():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(values):
        dtc = DecisionTreeClassifier(max_depth=values[0], min_samples_split=values[1])
        scores = cross_val_score(dtc, X, y, cv=5)

        return scores.mean()

    search_space = [np.arange(1, 3), np.arange(2, 4)]

    c_time1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=300)
    diff_time1 = time.time() - c_time1

    c_time2 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=300, memory=False)
    diff_time2 = time.time() - c_time2

    assert diff_time1 < diff_time2 * 0.3


def test_memory_warm_start():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(values):
        dtc = DecisionTreeClassifier(max_depth=values[0], min_samples_split=values[1])
        scores = cross_val_score(dtc, X, y, cv=5)

        return scores.mean()

    search_space = [np.arange(1, 10), np.arange(2, 20)]

    c_time1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=300)
    diff_time1 = time.time() - c_time1

    memory = {
        "values": opt.values,
        "scores": opt.scores,
    }

    c_time2 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=300, memory=memory)
    diff_time2 = time.time() - c_time2

    assert diff_time2 < diff_time1 * 0.5

