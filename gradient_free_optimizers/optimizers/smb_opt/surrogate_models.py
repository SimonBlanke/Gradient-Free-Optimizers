# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor_
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor_
from sklearn.ensemble import (
    GradientBoostingRegressor as _GradientBoostingRegressor_,
)


class EnsembleRegressor:
    def __init__(self, estimators, min_std=0.001):
        self.estimators = estimators
        self.min_std = min_std

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, np.ravel(y))

    def predict(self, X, return_std=False):
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X).reshape(-1, 1))

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        std[std < self.min_std] = self.min_std

        if return_std:

            return mean, std
        return mean


def _return_std(X, trees, predictions, min_variance):
    """
    used from:
    https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py
    """
    variance = np.zeros(len(X))
    trees = list(trees)

    for tree in trees:
        if isinstance(tree, np.ndarray):
            tree = tree[0]

        var_tree = tree.tree_.impurity[tree.apply(X)]
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        variance += var_tree ** 2 + mean_tree ** 2

    variance /= len(trees)
    variance -= predictions ** 2.0
    variance[variance < 0.0] = 0.0
    std = variance ** 0.5
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
            std = _return_std(X, self.estimators_, mean, self.min_variance)

            return mean, std
        return mean


class RandomForestRegressor(TreeEnsembleBase, _RandomForestRegressor_):
    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(min_variance=min_variance, **kwargs)


class ExtraTreesRegressor(TreeEnsembleBase, _ExtraTreesRegressor_):
    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(min_variance=min_variance, **kwargs)


class GradientBoostingRegressor(TreeEnsembleBase, _GradientBoostingRegressor_):
    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(min_variance=min_variance, **kwargs)


class GPR:
    def __init__(self):
        length_scale_param = 1
        length_scale_bounds_param = (1e-05, 100000.0)
        nu_param = 0.5
        matern = Matern(
            # length_scale=length_scale_param,
            # length_scale_bounds=length_scale_bounds_param,
            nu=nu_param,
        )

        self.gpr = GaussianProcessRegressor(
            kernel=matern + WhiteKernel(), n_restarts_optimizer=0
        )

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)


class GPR_linear:
    def __init__(self):
        self.gpr = BayesianRidge()

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)
