# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor_
from sklearn.ensemble import ExtraTreesRegressor as _RandomForestRegressor_


def _return_std(X, trees, predictions, min_variance):
    std = np.zeros(len(X))

    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)]
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree ** 2

    std /= len(trees)
    std -= predictions ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    # print("std", std)
    return std


class TreeEnsembleBase:
    def __init__(self, min_variance=0.0, **kwargs):
        self.min_variance = min_variance
        super().__init__(**kwargs)

    def fit(self, X, y):
        super().fit(X, np.ravel(y))

    def predict(self, X, return_std=False):
        mean = super().predict(X)

        if return_std:
            if self.criterion != "mse":
                raise ValueError(
                    "Expected impurity to be 'mse', got %s instead" % self.criterion
                )
            std = _return_std(X, self.estimators_, mean, self.min_variance)
            return mean, std
        return mean


class RandomForestRegressor(TreeEnsembleBase, _RandomForestRegressor_):
    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(**kwargs)


class ExtraTreesRegressor(TreeEnsembleBase, _ExtraTreesRegressor_):
    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(**kwargs)


class GPR:
    def __init__(self):
        self.gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)


class GPR_linear:
    def __init__(self):
        self.gpr = BayesianRidge(n_iter=10, normalize=True)

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)
