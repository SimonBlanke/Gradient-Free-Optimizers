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


def test_hyperactive():
    hyper = Hyperactive(X, y)
    hyper.add_search(model, search_space)
    hyper.run()

