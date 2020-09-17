import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from gradient_free_optimizers import RandomSearchOptimizer


def test_function():
    def objective_function(pos_new):
        score = -pos_new[0] * pos_new[0]
        return score

    search_space = [np.arange(-100, 101, 1)]

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)


def test_sklearn():
    data = load_iris()
    X, y = data.data, data.target

    def model(array):
        knr = KNeighborsClassifier(n_neighbors=array[0])
        scores = cross_val_score(knr, X, y, cv=5)
        score = scores.mean()

        return score

    search_space = [np.arange(1, 51, 1)]

    opt = RandomSearchOptimizer(search_space)
    opt.search(model, n_iter=30)

