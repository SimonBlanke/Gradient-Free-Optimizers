# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Kernel Density Estimator implementation.

Isotropic-bandwidth Gaussian Kernel Density Estimator for TPE.

This implementation uses the GFO array/math backend, enabling it to work
without NumPy/SciPy when needed (though it will be slower).
"""

import math

from gradient_free_optimizers._array_backend import (
    HAS_NUMPY,
    array,
    asarray,
    full,
    zeros,
)
from gradient_free_optimizers._math_backend import logsumexp

if HAS_NUMPY:
    import numpy as np


def _rows_as_tuples(X):
    """Convert 2D array-like to list of tuples for math.dist."""
    if hasattr(X, "tolist"):
        data = X.tolist()
    elif hasattr(X, "_data"):
        data = X._data if (hasattr(X, "ndim") and X.ndim == 2) else [X._data]
    else:
        data = list(X)
    if data and not isinstance(data[0], list | tuple):
        return [(v,) for v in data]
    return [tuple(row) for row in data]


class KernelDensityEstimator:
    """
    Isotropic-bandwidth Gaussian Kernel Density Estimator.

    Parameters
    ----------
    bandwidth : float | None, default=None
        The smoothing parameter 'h'. If None, Silverman's rule of thumb
        is applied during `fit`.
    atol : float, default=1e-12
        Absolute tolerance to truncate extremely small kernel
        contributions for numerical stability (used only in the
        exponential domain).
    """

    def __init__(self, bandwidth: float | None = None, atol: float = 1e-12):
        self._bandwidth_param = bandwidth
        self.bandwidth = bandwidth
        self.atol = atol
        self._fitted = False

    def fit(self, X) -> "KernelDensityEstimator":
        """Memorise the training samples and estimate bandwidth if necessary.

        Parameters
        ----------
        X : (n_samples, n_features) ndarray
            Training data.

        Returns
        -------
        self : KernelDensityEstimator
            Reference to the estimator (scikit-learn style).
        """
        X = asarray(X, dtype=float)

        # Handle 1D arrays
        if X.ndim == 1:
            # Reshape to 2D
            self.X_train = array([[x] for x in X])
        else:
            self.X_train = X

        self.n_samples, self.n_features = self.X_train.shape

        # Handle empty training data
        if self.n_samples == 0:
            self._fitted = True
            return self

        # Reset to original param so Silverman's rule recomputes on refit
        self.bandwidth = self._bandwidth_param

        if self.bandwidth is None:
            # Silverman's rule (vectorised across dimensions)
            # Compute std for each feature
            stds = []
            for j in range(self.n_features):
                col = [float(self.X_train[i, j]) for i in range(self.n_samples)]
                mean_val = sum(col) / len(col)
                variance = (
                    sum((x - mean_val) ** 2 for x in col) / (len(col) - 1)
                    if len(col) > 1
                    else 0
                )
                stds.append(math.sqrt(variance) if variance > 0 else 0)

            sigma = sum(stds) / len(stds) if stds else 0
            factor = (4 / (self.n_features + 2)) ** (1 / (self.n_features + 4))
            self.bandwidth = (
                factor * sigma * self.n_samples ** (-1 / (self.n_features + 4))
            )
            # Degenerate data (all identical points) produces zero bandwidth
            if self.bandwidth <= 0:
                self.bandwidth = 1.0
        elif self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        d = self.n_features
        self._log_norm_const = -0.5 * d * math.log(2 * math.pi) - d * math.log(
            self.bandwidth
        )
        self._fitted = True
        return self

    def score_samples(self, X, log: bool = True):
        """
        Evaluate (log-)density for each query point.

        Parameters
        ----------
        X : (n_queries, n_features) ndarray
            Locations where the density should be estimated.
        log : bool, default=True
            If True, return log-density; otherwise return density.

        Returns
        -------
        density : (n_queries,) ndarray
            Estimated (log-)density values.
        """
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before calling score_samples.")

        X = asarray(X, dtype=float)

        # Handle 1D arrays
        if X.ndim == 1:
            X = array([[x] for x in X])

        n_queries = len(X)

        # Handle empty training data - return -inf for log density (0 for density)
        if self.n_samples == 0:
            if log:
                return full(n_queries, float("-inf"))
            else:
                return zeros(n_queries)

        log_n = math.log(self.n_samples)

        if HAS_NUMPY:
            X_np = np.asarray(X, dtype=float)
            Xt_np = np.asarray(self.X_train, dtype=float)
            inv_bw_sq = 1.0 / (self.bandwidth * self.bandwidth)

            X_sq = np.sum(X_np * X_np, axis=1)
            Xt_sq = np.sum(Xt_np * Xt_np, axis=1)
            sq_dist = (
                X_sq[:, None] + Xt_sq[None, :] - 2.0 * (X_np @ Xt_np.T)
            ) * inv_bw_sq
            np.maximum(sq_dist, 0.0, out=sq_dist)

            log_kernels = self._log_norm_const - 0.5 * sq_dist
            a_max = np.max(log_kernels, axis=1, keepdims=True)
            log_sum = a_max.ravel() + np.log(
                np.sum(np.exp(log_kernels - a_max), axis=1)
            )
            log_density = log_sum - log_n

            if log:
                return array(log_density.tolist())
            density = np.exp(log_density)
            density[density < self.atol] = 0.0
            return array(density.tolist())

        X_rows = _rows_as_tuples(X)
        Xt_rows = _rows_as_tuples(self.X_train)
        _exp = math.exp
        _dist = math.dist
        inv_bw_sq = 1.0 / (self.bandwidth * self.bandwidth)

        log_density_values = []
        for x_row in X_rows:
            log_kernels = []
            for xt_row in Xt_rows:
                d = _dist(x_row, xt_row)
                log_kernels.append(self._log_norm_const - 0.5 * d * d * inv_bw_sq)
            log_density_values.append(logsumexp(log_kernels) - log_n)

        if log:
            return array(log_density_values)
        density = []
        for ld in log_density_values:
            d = _exp(ld)
            density.append(0.0 if d < self.atol else d)
        return array(density)
