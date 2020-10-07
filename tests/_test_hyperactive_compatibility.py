import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_iris()
X, y = data.data, data.target


def model(para, X, y):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"], min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(dtc, X, y, cv=2)

    return scores.mean()


search_space = {
    "max_depth": list(range(1, 21)),
    "min_samples_split": list(range(2, 21)),
}


def model1(para, X, y):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"], min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(dtc, X, y, cv=3)

    return scores.mean()


search_space1 = {
    "max_depth": list(range(1, 21)),
    "min_samples_split": list(range(2, 21)),
}


n_iter = 30


def test_hyperactive_0():
    hyper = Hyperactive(X, y)
    hyper.add_search(model, search_space, n_iter=n_iter)
    hyper.run()


def test_hyperactive_1():
    hyper = Hyperactive(X, y)
    hyper.add_search(model, search_space, n_iter=n_iter, n_jobs=2)
    hyper.run()


def test_hyperactive_2():
    hyper = Hyperactive(X, y)
    hyper.add_search(model, search_space, n_iter=n_iter, n_jobs=4)
    hyper.run()


def test_hyperactive_3():
    hyper = Hyperactive(X, y)
    hyper.add_search(model, search_space, n_iter=n_iter)
    hyper.add_search(model, search_space, n_iter=n_iter)
    hyper.run()


def test_hyperactive_4():
    hyper = Hyperactive(X, y)
    hyper.add_search(model, search_space, n_iter=n_iter)
    hyper.add_search(model1, search_space1, n_iter=n_iter)
    hyper.run()

