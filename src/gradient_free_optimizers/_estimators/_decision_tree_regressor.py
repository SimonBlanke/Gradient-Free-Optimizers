# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Decision Tree Regressor implementation.

A simple CART-style regression tree that is compatible with sklearn's interface.
"""

import numpy as np


class TreeNode:
    """Internal node structure for the decision tree."""

    __slots__ = [
        "feature",
        "threshold",
        "left",
        "right",
        "value",
        "impurity",
        "n_samples",
    ]

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.impurity = None
        self.n_samples = None


class TreeStructure:
    """
    Mimics sklearn's tree_ attribute for compatibility with _return_std().

    Attributes
    ----------
    impurity : ndarray
        Impurity (variance) at each node.
    """

    def __init__(self):
        self.impurity = None
        self._node_to_index = {}


class DecisionTreeRegressor:
    """
    Native Decision Tree Regressor.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int, float, or None, default=None
        Number of features to consider for best split.
        - None: use all features
        - int: use exactly that many features
        - float: use fraction of features
    random_state : int or None, default=None
        Random seed for reproducibility (used when max_features < n_features).
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self._root = None
        self._n_features = None
        self._rng = None
        self.tree_ = TreeStructure()

    def fit(self, X, y):
        """
        Build a decision tree regressor from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, self._n_features = X.shape

        # Setup random state
        if self.random_state is not None:
            self._rng = np.random.RandomState(self.random_state)
        else:
            self._rng = np.random.RandomState()

        # Determine max_features
        if self.max_features is None:
            self._max_features = self._n_features
        elif isinstance(self.max_features, float):
            self._max_features = max(1, int(self.max_features * self._n_features))
        else:
            self._max_features = min(self.max_features, self._n_features)

        # Build tree
        self._nodes = []
        self._root = self._build_tree(X, y, depth=0)

        # Build tree_ structure for compatibility
        self._build_tree_structure()

        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        node = TreeNode()
        node.n_samples = len(y)
        node.value = np.mean(y)
        node.impurity = np.var(y) if len(y) > 0 else 0.0

        self._nodes.append(node)

        # Check stopping criteria
        if self._should_stop(X, y, depth):
            return node

        # Find best split
        feature, threshold = self._find_best_split(X, y)

        if feature is None:
            return node

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return node

        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _should_stop(self, X, y, depth):
        """Check if we should stop splitting."""
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(np.unique(y)) == 1:
            return True
        return False

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        current_impurity = np.var(y) * n_samples

        # Select features to consider
        if self._max_features < n_features:
            features = self._rng.choice(n_features, self._max_features, replace=False)
        else:
            features = range(n_features)

        for feature in features:
            values = X[:, feature]
            thresholds = np.unique(values)

            if len(thresholds) <= 1:
                continue

            # Use midpoints between unique values
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # Calculate impurity reduction (variance reduction)
                left_impurity = np.var(y[left_mask]) * n_left if n_left > 0 else 0
                right_impurity = np.var(y[right_mask]) * n_right if n_right > 0 else 0

                gain = current_impurity - left_impurity - right_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree_structure(self):
        """Build the tree_ structure for sklearn compatibility."""
        impurities = [node.impurity for node in self._nodes]
        self.tree_.impurity = np.array(impurities)

        # Map nodes to indices
        for i, node in enumerate(self._nodes):
            self.tree_._node_to_index[id(node)] = i

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
            Predicted values.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        """Predict for a single sample."""
        node = self._root
        while node.feature is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def apply(self, X):
        """
        Return the index of the leaf for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        leaf_indices : ndarray of shape (n_samples,)
            Index of the leaf node for each sample.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._apply_one(x) for x in X])

    def _apply_one(self, x):
        """Return leaf index for a single sample."""
        node = self._root
        while node.feature is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return self.tree_._node_to_index[id(node)]
