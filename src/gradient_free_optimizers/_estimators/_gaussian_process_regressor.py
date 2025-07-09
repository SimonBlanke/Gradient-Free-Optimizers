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

    def fit(self, X, y):
        """
        Fit the GP hyper-parameters and pre-compute quantities needed
        for prediction.

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
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).ravel()

        if self.optimize:
            self.theta = self._optimise_theta()

        K = self.kernel(self.X, self.X, self.theta)
        sigma_n2 = np.exp(2 * self.theta[-1])
        K += sigma_n2 * np.eye(len(self.X))

        self.L = cholesky(K, lower=True, check_finite=False)
        self.alpha = cho_solve((self.L, True), self.y, check_finite=False)

        return self

    def _log_marginal_likelihood(self, theta):
        K = self.kernel(self.X, self.X, theta)
        sigma_n2 = np.exp(2 * theta[-1])
        K += sigma_n2 * np.eye(len(self.X))

        L = cholesky(K, lower=True, check_finite=False)
        alpha = cho_solve((L, True), self.y, check_finite=False)

        lml = -0.5 * self.y @ alpha
        lml -= np.sum(np.log(np.diag(L)))
        lml -= 0.5 * len(self.X) * np.log(2 * np.pi)

        eps = 1e-5
        grad = np.empty_like(theta)
        for i in range(len(theta)):
            theta_eps = theta.copy()
            theta_eps[i] += eps
            K_eps = self.kernel(self.X, self.X, theta_eps)
            K_eps += np.exp(2 * theta_eps[-1]) * np.eye(len(self.X))
            L_eps = cholesky(K_eps, lower=True, check_finite=False)
            alpha_eps = cho_solve((L_eps, True), self.y, check_finite=False)
            lml_eps = -0.5 * self.y @ alpha_eps
            lml_eps -= np.sum(np.log(np.diag(L_eps)))
            lml_eps -= 0.5 * len(self.X) * np.log(2 * np.pi)
            grad[i] = (lml_eps - lml) / eps

        return -lml, -grad

    def _optimise_theta(self):
        """Maximise log-marginal likelihood starting from self.theta."""
        res = minimize(
            fun=lambda th: self._log_marginal_likelihood(th),
            x0=self.theta,
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=100, ftol=1e-6),
        )
        return res.x
