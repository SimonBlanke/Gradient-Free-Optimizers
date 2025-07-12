import numpy as np
from scipy.special import logsumexp


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

    def fit(self, X: np.ndarray) -> "KernelDensityEstimator":
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
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]  # promote 1-D → 2-D

        self.X_train = X
        self.n_samples, self.n_features = X.shape

        if self.bandwidth is None:
            # Silverman's rule (vectorised across dimensions)
            std = X.std(axis=0, ddof=1)
            sigma = std.mean()  # geometric/arith mean
            factor = (4 / (self.n_features + 2)) ** (1 / (self.n_features + 4))
            self.bandwidth = (
                factor * sigma * self.n_samples ** (-1 / (self.n_features + 4))
            )

        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        d = self.n_features
        self._log_norm_const = -0.5 * d * np.log(2 * np.pi) - d * np.log(self.bandwidth)
        self._fitted = True
        return self

    def score_samples(self, X: np.ndarray, log: bool = True) -> np.ndarray:
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

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]

        # Pairwise squared Euclidean distances
        diff = (X[:, None, :] - self.X_train[None, :, :]) / self.bandwidth
        sq_dist = np.sum(diff * diff, axis=-1)  # shape: (n_queries, n_samples)

        # Log-kernel values
        log_kernel = self._log_norm_const - 0.5 * sq_dist

        # Numerically stable summation via logsumexp
        log_density = logsumexp(log_kernel, axis=1) - np.log(self.n_samples)

        if log:
            return log_density
        else:
            # Avoid exponentiating extremely small values
            density = np.exp(log_density)
            density[density < self.atol] = 0.0
            return density
