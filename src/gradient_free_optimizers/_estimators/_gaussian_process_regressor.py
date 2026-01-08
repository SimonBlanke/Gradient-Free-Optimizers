# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Native Gaussian Process Regressor implementation.

A probabilistic regression model that provides uncertainty estimates.

This implementation uses the GFO array/math backend, enabling it to work
without NumPy/SciPy when needed (though it will be slower).
"""

import math

from gradient_free_optimizers._array_backend import (
    array,
    asarray,
    diag,
    log,
    maximum,
    sqrt,
)
from gradient_free_optimizers._array_backend import (
    sum as arr_sum,
)
from gradient_free_optimizers._math_backend import (
    cho_solve,
    cholesky,
    minimize,
    solve_triangular,
)


class GaussianProcessRegressor:
    """
    Gaussian Process regressor.

    Parameters
    ----------
    kernel : callable
        Function `k(X, X2, theta)` returning an (m, n) kernel matrix.
    theta : array, shape (p,)
        Initial hyper-parameters passed verbatim to `kernel`.
        For the default RBF kernel: theta = [log_lengthscale, log_sigma_f, log_sigma_n].
    optimize : bool, default True
        If True, maximise the log-marginal likelihood wrt `theta` during `fit`.
    """

    def __init__(self, kernel=None, theta=None, optimize=True):
        self.kernel = kernel or self._rbf_kernel
        if theta is not None:
            self.theta = asarray(theta, dtype=float)
        else:
            self.theta = array([math.log(1.0), math.log(1.0), math.log(0.1)])
        self.optimize = optimize
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        """Fit the GP hyper-parameters and pre-compute for prediction.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Training inputs.
        y : ndarray, shape (n,)
            Training targets.

        Returns
        -------
        self : fitted estimator.
        """
        self.X = asarray(X, dtype=float)
        y_arr = asarray(y, dtype=float)
        # Flatten if needed
        if hasattr(y_arr, "ravel"):
            self.y = y_arr.ravel()
        else:
            self.y = y_arr

        if self.optimize:
            self.theta = self._optimise_theta()

        K = self.kernel(self.X, self.X, self.theta)
        len(self.X)
        sigma_n2 = math.exp(
            2
            * float(
                self.theta[-1] if hasattr(self.theta, "__getitem__") else self.theta
            )
        )

        # K += sigma_n2 * I
        K_reg = self._add_diagonal(K, sigma_n2)

        self.L = cholesky(K_reg, lower=True)
        self.alpha = cho_solve((self.L, True), self.y)

        return self

    def _add_diagonal(self, K, value):
        """Add value to diagonal of matrix K."""
        n = len(K)
        # Convert to list of lists for manipulation
        if hasattr(K, "_data"):
            result = [list(row) for row in K._data]
        elif hasattr(K, "tolist"):
            result = K.tolist()
        else:
            result = [list(row) for row in K]

        for i in range(n):
            result[i][i] += value
        return array(result)

    def _log_marginal_likelihood(self, theta):
        """Compute negative log marginal likelihood and gradient."""
        theta = asarray(theta, dtype=float)
        n = len(self.X)

        K = self.kernel(self.X, self.X, theta)
        sigma_n2 = math.exp(
            2 * float(theta[-1] if hasattr(theta, "__getitem__") else theta)
        )
        K_reg = self._add_diagonal(K, sigma_n2)

        try:
            L = cholesky(K_reg, lower=True)
        except Exception:
            # Return high value if Cholesky fails
            return float("inf")

        alpha = cho_solve((L, True), self.y)

        # Compute log marginal likelihood
        # lml = -0.5 * y @ alpha - sum(log(diag(L))) - 0.5 * n * log(2*pi)
        y_alpha = self._dot_1d(self.y, alpha)
        L_diag = diag(L)
        log_det = float(arr_sum(log(L_diag)))

        lml = -0.5 * y_alpha - log_det - 0.5 * n * math.log(2 * math.pi)

        return -float(lml)  # Return negative for minimization

    def _dot_1d(self, a, b):
        """Dot product of two 1D arrays."""
        if hasattr(a, "_data"):
            a_list = a._data if a.ndim == 1 else a._get_flat()
        else:
            a_list = list(a)

        if hasattr(b, "_data"):
            b_list = b._data if b.ndim == 1 else b._get_flat()
        else:
            b_list = list(b)

        return sum(x * y for x, y in zip(a_list, b_list))

    def _optimise_theta(self):
        """Maximise log-marginal likelihood starting from self.theta."""
        # Convert theta to list for minimize
        if hasattr(self.theta, "tolist"):
            x0 = self.theta.tolist()
        elif hasattr(self.theta, "_data"):
            x0 = list(self.theta._data)
        else:
            x0 = list(self.theta)

        res = minimize(
            fun=self._log_marginal_likelihood,
            x0=x0,
            options={"maxiter": 100, "gtol": 1e-6},
        )
        return asarray(res.x, dtype=float)

    def predict(self, X_star, return_std=False):
        """
        Predict the GP posterior mean and (optionally) standard deviation.

        Parameters
        ----------
        X_star : ndarray, shape (m, d)
            Test inputs.
        return_std : bool, default False
            If True, also return predictive standard deviation.

        Returns
        -------
        mu : ndarray, shape (m,)
            Posterior mean.
        std : ndarray, shape (m,), optional
            Posterior standard deviation (present iff `return_std` is True).
        """
        X_star = asarray(X_star, dtype=float)

        K_starX = self.kernel(X_star, self.X, self.theta)

        # mu = K_starX @ alpha
        mu = self._matmul_mv(K_starX, self.alpha)

        if not return_std:
            return mu

        # v = L^{-1} @ K_starX.T  (solve L @ v = K_starX.T)
        K_starX_T = self._transpose(K_starX)
        v = solve_triangular(self.L, K_starX_T, lower=True)

        # K_starstar = kernel diagonal
        K_starstar = self.kernel(X_star, X_star, self.theta, diag=True)

        # var = K_starstar - sum(v**2, axis=0)
        v_squared_sum = self._sum_squared_cols(v)
        var = self._subtract_arrays(K_starstar, v_squared_sum)

        # std = sqrt(max(var, 0))
        std = sqrt(maximum(var, 0))

        return mu, std

    def _transpose(self, A):
        """Transpose a 2D array."""
        if hasattr(A, "T"):
            return A.T
        # Manual transpose
        if hasattr(A, "_data"):
            data = A._data
        else:
            data = list(A)
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        result = [[data[i][j] for i in range(rows)] for j in range(cols)]
        return array(result)

    def _matmul_mv(self, A, v):
        """Matrix-vector multiplication."""
        if hasattr(A, "__matmul__"):
            return A @ v

        # Manual implementation
        if hasattr(A, "_data"):
            A_data = A._data
        else:
            A_data = list(A)

        if hasattr(v, "_data"):
            v_data = v._data if v.ndim == 1 else v._get_flat()
        else:
            v_data = list(v)

        result = []
        for row in A_data:
            result.append(sum(a * b for a, b in zip(row, v_data)))
        return array(result)

    def _sum_squared_cols(self, v):
        """Sum of squared values along columns (axis=0)."""
        if hasattr(v, "_data"):
            data = v._data
        else:
            data = list(v)

        if not data:
            return array([])

        # v is (n, m), we want sum over rows for each column
        n_rows = len(data)
        n_cols = len(data[0]) if n_rows > 0 else 0

        result = []
        for j in range(n_cols):
            col_sum = sum(data[i][j] ** 2 for i in range(n_rows))
            result.append(col_sum)
        return array(result)

    def _subtract_arrays(self, a, b):
        """Element-wise subtraction."""
        if hasattr(a, "__sub__"):
            return a - b

        if hasattr(a, "_data"):
            a_list = a._data if a.ndim == 1 else a._get_flat()
        else:
            a_list = list(a)

        if hasattr(b, "_data"):
            b_list = b._data if b.ndim == 1 else b._get_flat()
        else:
            b_list = list(b)

        return array([x - y for x, y in zip(a_list, b_list)])

    @staticmethod
    def _rbf_kernel(X1, X2, theta, diag=False):
        """
        RBF (squared exponential) kernel.

        Parameters
        ----------
        X1, X2 : arrays
            Input arrays
        theta : array
            [log_lengthscale, log_sigma_f, log_sigma_n]
        diag : bool
            If True, return only diagonal (for variance computation)
        """
        # Extract hyperparameters
        if hasattr(theta, "__getitem__"):
            log_ell = float(theta[0])
            log_sigma_f = float(theta[1])
        else:
            log_ell = float(theta)
            log_sigma_f = 0.0

        ell = math.exp(log_ell)
        sigma_f2 = math.exp(2 * log_sigma_f)

        X1 = asarray(X1, dtype=float)
        X2 = asarray(X2, dtype=float)

        # Get data as nested Python lists (pure floats, not numpy types)
        if hasattr(X1, "_data"):
            X1_data = X1._data if X1.ndim == 2 else [X1._data]
        elif hasattr(X1, "tolist"):
            # numpy array - convert to pure Python lists
            X1_data = X1.tolist()
            if not isinstance(X1_data[0], list):
                X1_data = [[x] for x in X1_data]
        else:
            X1_data = list(X1)
            if not isinstance(X1_data[0], list | tuple):
                X1_data = [[x] for x in X1_data]

        if hasattr(X2, "_data"):
            X2_data = X2._data if X2.ndim == 2 else [X2._data]
        elif hasattr(X2, "tolist"):
            # numpy array - convert to pure Python lists
            X2_data = X2.tolist()
            if not isinstance(X2_data[0], list):
                X2_data = [[x] for x in X2_data]
        else:
            X2_data = list(X2)
            if not isinstance(X2_data[0], list | tuple):
                X2_data = [[x] for x in X2_data]

        n1 = len(X1_data)
        n2 = len(X2_data)

        if diag:
            # Return diagonal only (all ones scaled by sigma_f2 for same points)
            return array([sigma_f2] * n1)

        # Compute full kernel matrix
        # K[i,j] = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / ell^2)
        result = []
        for i in range(n1):
            row = []
            for j in range(n2):
                # Squared distance
                sq_dist = 0.0
                for k in range(len(X1_data[i])):
                    diff = (X1_data[i][k] - X2_data[j][k]) / ell
                    sq_dist += diff * diff
                k_val = sigma_f2 * math.exp(-0.5 * sq_dist)
                row.append(k_val)
            result.append(row)

        return array(result)
