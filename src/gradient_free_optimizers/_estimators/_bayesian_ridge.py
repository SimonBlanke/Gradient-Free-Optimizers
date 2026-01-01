# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Bayesian Ridge Regression implementation.

Linear regression with Bayesian regularization and uncertainty estimates.
"""

import numpy as np
from scipy.linalg import solve


class BayesianRidge:
    """
    Native Bayesian Ridge Regression.

    Fits a linear model with automatic regularization and provides
    uncertainty estimates for predictions.

    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations for hyperparameter optimization.
    tol : float, default=1e-3
        Convergence threshold for stopping criterion.
    alpha_1 : float, default=1e-6
        Shape parameter for Gamma prior over alpha (noise precision).
    alpha_2 : float, default=1e-6
        Rate parameter for Gamma prior over alpha.
    lambda_1 : float, default=1e-6
        Shape parameter for Gamma prior over lambda (weight precision).
    lambda_2 : float, default=1e-6
        Rate parameter for Gamma prior over lambda.
    compute_score : bool, default=False
        If True, compute log marginal likelihood at each iteration.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    """

    def __init__(
        self,
        n_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=False,
        fit_intercept=True,
    ):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.intercept_ = 0.0
        self.alpha_ = None  # Noise precision
        self.lambda_ = None  # Weight precision
        self.sigma_ = None  # Posterior covariance

    def fit(self, X, y):
        """
        Fit the Bayesian Ridge model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BayesianRidge
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X = X - X_mean
            y = y - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Precompute X^T X and X^T y
        XtX = X.T @ X
        Xty = X.T @ y

        # Initialize hyperparameters
        alpha_ = 1.0 / np.var(y) if np.var(y) > 0 else 1.0
        lambda_ = 1.0

        # Eigendecomposition for efficient updates
        eigenvalues = np.linalg.eigvalsh(XtX)

        # Iterative optimization
        for iteration in range(self.n_iter):
            # Posterior precision and covariance
            A = lambda_ * np.eye(n_features) + alpha_ * XtX

            try:
                # Solve for posterior mean
                coef_ = solve(A, alpha_ * Xty, assume_a='pos')
                # Posterior covariance
                sigma_ = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                coef_ = np.linalg.lstsq(A, alpha_ * Xty, rcond=None)[0]
                sigma_ = np.linalg.pinv(A)

            # Update alpha (noise precision)
            residuals = y - X @ coef_
            sse = np.sum(residuals ** 2)
            gamma = np.sum(alpha_ * eigenvalues / (lambda_ + alpha_ * eigenvalues))

            alpha_new = (n_samples - gamma + 2 * self.alpha_1) / (
                sse + 2 * self.alpha_2
            )

            # Update lambda (weight precision)
            lambda_new = (gamma + 2 * self.lambda_1) / (
                np.sum(coef_ ** 2) + 2 * self.lambda_2
            )

            # Check convergence
            if (
                np.abs(alpha_new - alpha_) < self.tol
                and np.abs(lambda_new - lambda_) < self.tol
            ):
                break

            alpha_ = alpha_new
            lambda_ = lambda_new

        self.coef_ = coef_
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.sigma_ = sigma_

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X, return_std=False):
        """
        Predict using the Bayesian Ridge model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.
        return_std : bool, default=False
            If True, return standard deviation of predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        y_std : ndarray of shape (n_samples,), optional
            Standard deviation of predictions (if return_std=True).
        """
        X = np.asarray(X, dtype=np.float64)

        y_pred = X @ self.coef_ + self.intercept_

        if return_std:
            # Predictive variance = noise variance + parameter uncertainty
            var_pred = 1.0 / self.alpha_ + np.sum(X @ self.sigma_ * X, axis=1)
            y_std = np.sqrt(np.maximum(var_pred, 0))
            return y_pred, y_std

        return y_pred
