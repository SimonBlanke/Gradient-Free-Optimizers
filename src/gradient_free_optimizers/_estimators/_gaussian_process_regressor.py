import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize


class GaussianProcessRegressor:
    """
    Gaussian Process regressor.

    Parameters
    ----------
    kernel : callable
        Function `k(X, X2, theta)` returning an (m, n) kernel matrix.
    theta : ndarray, shape (p,)
        Initial hyper-parameters passed verbatim to `kernel`.
        For the default RBF kernel: theta = [log_lengthscale, log_sigma_f, log_sigma_n].
    optimize : bool, default True
        If True, maximise the log-marginal likelihood wrt `theta` during `fit`.
    """

    def __init__(self, kernel=None, theta=None, optimize=True):
        self.kernel = kernel or self._rbf_kernel
        self.theta = np.asarray(
            theta if theta is not None else np.log([1.0, 1.0, 1e-1])
        )
        self.optimize = optimize
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None
