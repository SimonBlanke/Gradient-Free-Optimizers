# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Extra Trees Regressor implementation.

Extremely randomized trees - uses random splits instead of optimal splits.

This implementation uses the GFO array backend, enabling it to work
without NumPy when needed (though it will be slower).
"""

import math

from gradient_free_optimizers._array_backend import (
    array,
    asarray,
    var,
)
from gradient_free_optimizers._array_backend import (
    random as np_random,
)
from gradient_free_optimizers._array_backend import (
    sum as arr_sum,
)

from ._decision_tree_regressor import DecisionTreeRegressor


class ExtraTreeRegressor(DecisionTreeRegressor):
    """
    A single extremely randomized tree.

    Uses random thresholds instead of searching for optimal splits.
    """

    def _find_best_split(self, X, y):
        """Find a random split (instead of optimal)."""
        n_samples, n_features = X.shape

        if n_samples < 2:
            return None, None

        # Select features to consider
        if self._max_features < n_features:
            features = self._rng.choice(n_features, self._max_features, replace=False)
            features = list(features) if hasattr(features, "__iter__") else [features]
        else:
            features = range(n_features)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        current_impurity = float(var(y)) * n_samples

        for feature in features:
            values = X[:, feature]
            min_val, max_val = float(values.min()), float(values.max())

            if min_val == max_val:
                continue

            # Random threshold between min and max
            threshold = self._rng.uniform(min_val, max_val)

            left_mask = values <= threshold
            right_mask = ~left_mask

            n_left = int(arr_sum(left_mask))
            n_right = int(arr_sum(right_mask))

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            # Calculate impurity reduction
            left_impurity = float(var(y[left_mask])) * n_left if n_left > 0 else 0
            right_impurity = float(var(y[right_mask])) * n_right if n_right > 0 else 0

            gain = current_impurity - left_impurity - right_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = float(threshold)

        return best_feature, best_threshold


class ExtraTreesRegressor:
    """
    Native Extra Trees Regressor.

    An ensemble of extremely randomized trees.

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
        Number of features to consider for splits.
    bootstrap : bool, default=False
        Whether to use bootstrap samples (False for Extra Trees).
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
        bootstrap=False,
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
        Build a forest of extremely randomized trees.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ExtraTreesRegressor
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
            tree = ExtraTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=rng.randint(0, 2**31 - 1),
            )

            # Bootstrap sample (usually False for Extra Trees)
            if self.bootstrap:
                indices = rng.randint(0, n_samples, n_samples)
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
