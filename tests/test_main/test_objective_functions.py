import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from gradient_free_optimizers import RandomSearchOptimizer


def test_function():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)


def test_sklearn():
    data = load_iris()
    X, y = data.data, data.target

    def model(para):
        knr = KNeighborsClassifier(n_neighbors=para["n_neighbors"])
        scores = cross_val_score(knr, X, y, cv=5)
        score = scores.mean()

        return score

    search_space = {
        "n_neighbors": np.arange(1, 51, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(model, n_iter=30)


def test_obj_func_return_dictionary_0():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score, {"_x1_": para["x1"]}

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)

    assert "_x1_" in list(opt.search_data.columns)


def test_obj_func_return_dictionary_1():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score, {"_x1_": para["x1"], "_x1_*2": para["x1"] * 2}

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)

    assert "_x1_" in list(opt.search_data.columns)
    assert "_x1_*2" in list(opt.search_data.columns)
