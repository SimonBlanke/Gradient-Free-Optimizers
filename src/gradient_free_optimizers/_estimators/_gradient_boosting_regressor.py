# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Gradient Boosting Regressor implementation.

Sequential ensemble where each tree corrects the errors of the previous ones.
"""

import numpy as np
from ._decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingRegressor:
    """
    Native Gradient Boosting Regressor.

    Builds trees sequentially, each fitting the residuals of the previous.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    subsample : float, default=1.0
        Fraction of samples used for fitting each tree.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

        self.estimators_ = []
        self._init_prediction = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GradientBoostingRegressor
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples = X.shape[0]

        # Setup random state
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()

        # Initialize with mean
        self._init_prediction = np.mean(y)
        current_prediction = np.full(n_samples, self._init_prediction)

        # Build trees sequentially
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for MSE loss)
            residuals = y - current_prediction

            # Subsample
            if self.subsample < 1.0:
                n_subsample = max(1, int(self.subsample * n_samples))
                indices = rng.choice(n_samples, n_subsample, replace=False)
                X_sample = X[indices]
                residuals_sample = residuals[indices]
            else:
                X_sample = X
                residuals_sample = residuals

            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 2**31 - 1),
            )
            tree.fit(X_sample, residuals_sample)

            # Update predictions
            update = self.learning_rate * tree.predict(X)
            current_prediction += update

            # Store as nested array for sklearn compatibility
            self.estimators_.append(np.array([tree]))

        return self

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

        prediction = np.full(X.shape[0], self._init_prediction)

        for tree_array in self.estimators_:
            tree = tree_array[0]
            prediction += self.learning_rate * tree.predict(X)

        return prediction
