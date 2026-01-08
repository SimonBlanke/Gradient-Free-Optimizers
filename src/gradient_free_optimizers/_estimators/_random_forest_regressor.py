# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Random Forest Regressor implementation.

An ensemble of decision trees with bootstrap sampling and feature subsampling.

This implementation uses the GFO array backend, enabling it to work
without NumPy when needed (though it will be slower).
"""

import math

from gradient_free_optimizers._array_backend import (
    array,
    asarray,
)
from gradient_free_optimizers._array_backend import (
    random as np_random,
)

from ._decision_tree_regressor import DecisionTreeRegressor


class RandomForestRegressor:
    """
    Native Random Forest Regressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    max_features : str, int, float, or None, default="sqrt"
        Number of features to consider for best split:
        - "sqrt": sqrt(n_features)
        - "log2": log2(n_features)
        - int: exact number
        - float: fraction
        - None: all features
    bootstrap : bool, default=True
        Whether to use bootstrap samples.
    random_state : int or None, default=None
        Random seed for reproducibility.
    n_jobs : int or None, default=None
        Not used, kept for sklearn compatibility.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
        n_jobs=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimators_ = []

    def fit(self, X, y):
        """
        Build a forest of trees from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForestRegressor
            Fitted estimator.
        """
        X = asarray(X, dtype=float)
        y = asarray(y, dtype=float).ravel()

        n_samples, n_features = X.shape

        # Setup random state
        rng = np_random.RandomState(self.random_state)

        # Determine max_features
        max_features = self._get_max_features(n_features)

        # Build trees
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Create tree with unique random state
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=rng.randint(0, 2**31 - 1),
            )

            # Bootstrap sample
            if self.bootstrap:
                indices = rng.randint(0, n_samples, n_samples)
                # Convert indices to list for indexing
                indices_list = (
                    list(indices) if hasattr(indices, "__iter__") else [indices]
                )
                X_sample = array([X[i] for i in indices_list])
                y_sample = array([y[i] for i in indices_list])
            else:
                X_sample = X
                y_sample = y

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def _get_max_features(self, n_features):
        """Convert max_features parameter to integer."""
        if self.max_features is None:
            return n_features
        elif self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(math.log2(n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            return min(self.max_features, n_features)

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (mean of all trees).
        """
        X = asarray(X, dtype=float)

        # Collect predictions from all trees and average
        all_preds = [tree.predict(X) for tree in self.estimators_]
        n_samples = len(X)
        result = []
        for i in range(n_samples):
            sample_preds = [float(preds[i]) for preds in all_preds]
            result.append(sum(sample_preds) / len(sample_preds))
        return array(result)
