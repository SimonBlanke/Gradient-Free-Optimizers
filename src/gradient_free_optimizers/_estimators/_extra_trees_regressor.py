# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Extra Trees Regressor implementation.

Extremely randomized trees - uses random splits instead of optimal splits.
"""

import numpy as np
from ._decision_tree_regressor import DecisionTreeRegressor, TreeNode


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
        else:
            features = range(n_features)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        current_impurity = np.var(y) * n_samples

        for feature in features:
            values = X[:, feature]
            min_val, max_val = values.min(), values.max()

            if min_val == max_val:
                continue

            # Random threshold between min and max
            threshold = self._rng.uniform(min_val, max_val)

            left_mask = values <= threshold
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            # Calculate impurity reduction
            left_impurity = np.var(y[left_mask]) * n_left if n_left > 0 else 0
            right_impurity = np.var(y[right_mask]) * n_right if n_right > 0 else 0

            gain = current_impurity - left_impurity - right_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

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
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        # Setup random state
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()

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
                X_sample = X[indices]
                y_sample = y[indices]
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
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
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
        X = np.asarray(X, dtype=np.float64)

        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)
