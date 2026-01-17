# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate models for Sequential Model-Based Optimization.

Uses sklearn implementations if available, otherwise falls back to native
implementations that have no external dependencies beyond numpy/scipy.
"""

from gradient_free_optimizers._array_backend import array, ravel
from gradient_free_optimizers._estimators import (
    ExtraTreesRegressor as NativeExtraTreesRegressor,
)

# Import native implementations as fallbacks
from gradient_free_optimizers._estimators import (
    GaussianProcessRegressor as NativeGPR,
)
from gradient_free_optimizers._estimators import (
    GradientBoostingRegressor as NativeGradientBoostingRegressor,
)
from gradient_free_optimizers._estimators import (
    RandomForestRegressor as NativeRandomForestRegressor,
)

# Try to import sklearn implementations
try:
    from sklearn.ensemble import ExtraTreesRegressor as SklearnExtraTreesRegressor
    from sklearn.ensemble import (
        GradientBoostingRegressor as SklearnGradientBoostingRegressor,
    )
    from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor as SklearnGPR
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EnsembleRegressor:
    """Ensemble of multiple regressors with uncertainty from variance."""

    def __init__(self, estimators, min_std=0.001):
        self.estimators = estimators
        self.min_std = min_std

    def fit(self, X, y):
        y_flat = ravel(array(y))
        for estimator in self.estimators:
            estimator.fit(X, y_flat)

    def predict(self, X, return_std=False):
        predictions = []
        for estimator in self.estimators:
            pred = array(estimator.predict(X)).reshape(-1, 1)
            predictions.append(pred)

        predictions = array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        # Set minimum std
        for i in range(len(std)):
            if std[i] < self.min_std:
                std[i] = self.min_std

        if return_std:
            return mean, std
        return mean


def _return_std(X, trees, predictions, min_variance):
    """
    Compute standard deviation from tree ensemble variance.

    Used from:
    https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py

    Note: This function requires sklearn trees with tree_ attribute,
    so we keep numpy usage here for sklearn integration.
    """
    import numpy as np

    variance = np.zeros(len(X))
    trees = list(trees)

    for tree in trees:
        if isinstance(tree, np.ndarray):
            tree = tree[0]

        var_tree = tree.tree_.impurity[tree.apply(X)]
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        variance += var_tree**2 + mean_tree**2

    variance /= len(trees)
    variance -= predictions**2.0
    variance[variance < 0.0] = 0.0
    std = variance**0.5
    return std


class TreeEnsembleBase:
    """Base class for tree ensembles with uncertainty estimation."""

    def __init__(self, min_variance=0.0):
        self.min_variance = min_variance

    def fit(self, X, y):
        y_flat = ravel(array(y))
        self._estimator.fit(X, y_flat)

    def predict(self, X, return_std=False):
        mean = self._estimator.predict(X)

        if return_std:
            std = _return_std(X, self._estimator.estimators_, mean, self.min_variance)
            return mean, std
        return mean


class RandomForestRegressor(TreeEnsembleBase):
    """Random Forest Regressor with uncertainty estimation."""

    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(min_variance=min_variance)
        if SKLEARN_AVAILABLE:
            self._estimator = SklearnRandomForestRegressor(**kwargs)
        else:
            self._estimator = NativeRandomForestRegressor(**kwargs)


class ExtraTreesRegressor(TreeEnsembleBase):
    """Extra Trees Regressor with uncertainty estimation."""

    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(min_variance=min_variance)
        if SKLEARN_AVAILABLE:
            self._estimator = SklearnExtraTreesRegressor(**kwargs)
        else:
            self._estimator = NativeExtraTreesRegressor(**kwargs)


class GradientBoostingRegressor(TreeEnsembleBase):
    """Gradient Boosting Regressor with uncertainty estimation."""

    def __init__(self, min_variance=0.0, **kwargs):
        super().__init__(min_variance=min_variance)
        if SKLEARN_AVAILABLE:
            self._estimator = SklearnGradientBoostingRegressor(**kwargs)
        else:
            self._estimator = NativeGradientBoostingRegressor(**kwargs)


class GPR:
    """
    Gaussian Process Regressor.

    Uses sklearn's GP if available (with Matern + WhiteKernel),
    otherwise falls back to native RBF-based implementation.
    """

    def __init__(self):
        if SKLEARN_AVAILABLE:
            matern = Matern(nu=0.5)
            self.gpr = SklearnGPR(
                kernel=matern + WhiteKernel(),
                n_restarts_optimizer=0,
            )
        else:
            self.gpr = NativeGPR(optimize=True)

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)
