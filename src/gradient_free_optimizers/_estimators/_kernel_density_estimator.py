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

