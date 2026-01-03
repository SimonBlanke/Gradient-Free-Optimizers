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
    array,
    asarray,
    zeros,
    full,
)
from gradient_free_optimizers._math_backend import logsumexp


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
        self.bandwidth = bandwidth
        self.atol = atol
        self._fitted = False

    def fit(self, X) -> "KernelDensityEstimator":
        """
        Memorise the training samples and, if necessary, estimate a
        bandwidth.

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

        if self.bandwidth <= 0:
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

        # Compute log kernel values for each query-train pair
        log_density_values = []

        for i in range(n_queries):
            # Compute log kernel values for this query against all training samples
            log_kernels = []
            for j in range(self.n_samples):
                # Compute squared distance between query[i] and train[j]
                sq_dist = 0.0
                for k in range(self.n_features):
                    diff = (float(X[i, k]) - float(self.X_train[j, k])) / self.bandwidth
                    sq_dist += diff * diff

                log_kernel = self._log_norm_const - 0.5 * sq_dist
                log_kernels.append(log_kernel)

            # Numerically stable summation via logsumexp
            log_sum = logsumexp(log_kernels)
            log_density = log_sum - math.log(self.n_samples)
            log_density_values.append(log_density)

        if log:
            return array(log_density_values)
        else:
            # Avoid exponentiating extremely small values
            density = []
            for ld in log_density_values:
                d = math.exp(ld)
                density.append(0.0 if d < self.atol else d)
            return array(density)
