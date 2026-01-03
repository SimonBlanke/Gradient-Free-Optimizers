"""
Pure Python math functions - SciPy-compatible interface without dependencies.

This module provides pure Python implementations of mathematical functions
used by GFO that would normally come from SciPy.
"""

import math
from typing import Optional, Tuple, Union, List, Any

# Try to import numpy, fall back to pure array backend
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    from .._array_backend import _pure as np
    HAS_NUMPY = False


# === Statistical Functions ===

def norm_cdf(x, loc=0, scale=1):
    """
    Cumulative distribution function of normal distribution.

    Uses the error function approximation for numerical stability.
    """
    if hasattr(x, "__iter__"):
        return np.array([norm_cdf(xi, loc, scale) for xi in x])

    z = (x - loc) / scale
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def norm_pdf(x, loc=0, scale=1):
    """
    Probability density function of normal distribution.
    """
    if hasattr(x, "__iter__"):
        return np.array([norm_pdf(xi, loc, scale) for xi in x])

    z = (x - loc) / scale
    return math.exp(-0.5 * z * z) / (scale * math.sqrt(2 * math.pi))


def _erf(x):
    """
    Error function approximation.

    Uses Horner's method for the polynomial approximation.
    Maximum error: 1.5e-7
    """
    # Save sign
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # Abramowitz and Stegun approximation (equation 7.1.26)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


# === Linear Algebra ===

def cholesky(a, lower=True):
    """
    Cholesky decomposition of a positive-definite matrix.

    Returns lower triangular matrix L such that A = L @ L.T

    Note: This is a simplified implementation for small matrices.
    For production use with large matrices, scipy is recommended.
    """
    if HAS_NUMPY:
        a = np.asarray(a)
        n = a.shape[0]
        L = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i, k] * L[j, k] for k in range(j))

                if i == j:
                    val = a[i, i] - s
                    if val < 0:
                        raise ValueError("Matrix is not positive definite")
                    L[i, j] = math.sqrt(val)
                else:
                    if L[j, j] == 0:
                        L[i, j] = 0
                    else:
                        L[i, j] = (a[i, j] - s) / L[j, j]

        if lower:
            return L
        return L.T
    else:
        raise NotImplementedError(
            "Cholesky decomposition requires numpy. "
            "Install numpy or scipy for full functionality."
        )


def cho_solve(c_and_lower, b):
    """
    Solve using Cholesky decomposition.

    Solves A @ x = b given Cholesky decomposition L where A = L @ L.T
    """
    L, lower = c_and_lower

    if HAS_NUMPY:
        L = np.asarray(L)
        b = np.asarray(b)
        n = L.shape[0]

        # Forward substitution: L @ y = b
        y = np.zeros(n)
        for i in range(n):
            s = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - s) / L[i, i]

        # Backward substitution: L.T @ x = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            s = sum(L[j, i] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / L[i, i]

        return x
    else:
        raise NotImplementedError(
            "cho_solve requires numpy. Install numpy or scipy for full functionality."
        )


def solve(a, b, assume_a=None):
    """
    Solve linear system a @ x = b using Gaussian elimination.

    Note: This is a simplified implementation. For numerical stability
    with ill-conditioned matrices, scipy is recommended.
    """
    if HAS_NUMPY:
        a = np.asarray(a, dtype=float).copy()
        b = np.asarray(b, dtype=float).copy()
        n = a.shape[0]

        # Augmented matrix
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(a[k, i]) > abs(a[max_row, i]):
                    max_row = k

            # Swap rows
            a[[i, max_row]] = a[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

            # Eliminate column
            for k in range(i + 1, n):
                if a[i, i] != 0:
                    c = a[k, i] / a[i, i]
                    a[k, i:] -= c * a[i, i:]
                    b[k] -= c * b[i]

        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= a[i, j] * x[j]
            if a[i, i] != 0:
                x[i] /= a[i, i]

        return x
    else:
        raise NotImplementedError(
            "solve requires numpy. Install numpy or scipy for full functionality."
        )


def solve_triangular(a, b, lower=True):
    """
    Solve triangular linear system.

    Solves a @ x = b where a is triangular.

    Parameters
    ----------
    a : (n, n) array
        Triangular matrix (lower or upper)
    b : (n,) or (n, m) array
        Right-hand side vector(s)
    lower : bool
        True if a is lower triangular, False if upper triangular

    Returns
    -------
    x : array
        Solution vector(s)
    """
    if HAS_NUMPY:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        # Handle 1D b
        if b.ndim == 1:
            return _solve_triangular_1d(a, b, lower)

        # Handle 2D b (solve for each column)
        n, m = b.shape
        x = np.zeros((n, m))
        for j in range(m):
            x[:, j] = _solve_triangular_1d(a, b[:, j], lower)
        return x
    else:
        raise NotImplementedError(
            "solve_triangular requires numpy. Install numpy or scipy for full functionality."
        )


def _solve_triangular_1d(a, b, lower):
    """Solve triangular system for 1D right-hand side."""
    n = len(b)
    x = np.zeros(n)

    if lower:
        # Forward substitution
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= a[i, j] * x[j]
            x[i] = s / a[i, i]
    else:
        # Backward substitution
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= a[i, j] * x[j]
            x[i] = s / a[i, i]

    return x


# === Optimization ===

class OptimizeResult:
    """Simple result container for minimize."""
    def __init__(self, x, fun, success, message=""):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message


def minimize(fun, x0, method=None, bounds=None, options=None):
    """
    Minimize a function using simple gradient-free methods.

    Note: This is a very basic implementation using coordinate descent.
    For serious optimization, use scipy.optimize.minimize.
    """
    if HAS_NUMPY:
        x = np.asarray(x0, dtype=float).copy()
    else:
        x = list(x0)

    options = options or {}
    maxiter = options.get("maxiter", 100)
    tol = options.get("gtol", 1e-5)

    best_f = fun(x)
    step = 0.1

    for iteration in range(maxiter):
        improved = False

        for i in range(len(x)):
            # Try positive step
            x_new = x.copy() if HAS_NUMPY else list(x)
            x_new[i] += step

            # Apply bounds if given
            if bounds is not None:
                lo, hi = bounds[i]
                if lo is not None:
                    x_new[i] = max(lo, x_new[i])
                if hi is not None:
                    x_new[i] = min(hi, x_new[i])

            f_new = fun(x_new)
            if f_new < best_f:
                x = x_new
                best_f = f_new
                improved = True
                continue

            # Try negative step
            x_new = x.copy() if HAS_NUMPY else list(x)
            x_new[i] -= step

            if bounds is not None:
                lo, hi = bounds[i]
                if lo is not None:
                    x_new[i] = max(lo, x_new[i])
                if hi is not None:
                    x_new[i] = min(hi, x_new[i])

            f_new = fun(x_new)
            if f_new < best_f:
                x = x_new
                best_f = f_new
                improved = True

        if not improved:
            step *= 0.5
            if step < tol:
                break

    return OptimizeResult(
        x=x,
        fun=best_f,
        success=True,
        message="Optimization terminated successfully."
    )


# === Distance Functions ===

def cdist(xa, xb, metric="euclidean"):
    """
    Compute pairwise distances between two sets of points.

    Parameters
    ----------
    xa : array-like, shape (m, k)
        First set of points
    xb : array-like, shape (n, k)
        Second set of points
    metric : str
        Distance metric ("euclidean" supported)

    Returns
    -------
    distances : array, shape (m, n)
        Distance matrix
    """
    if HAS_NUMPY:
        xa = np.asarray(xa)
        xb = np.asarray(xb)

        if xa.ndim == 1:
            xa = xa.reshape(-1, 1)
        if xb.ndim == 1:
            xb = xb.reshape(-1, 1)

        m = xa.shape[0]
        n = xb.shape[0]

        if metric == "euclidean":
            result = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    diff = xa[i] - xb[j]
                    result[i, j] = math.sqrt(np.sum(diff * diff))
            return result
        else:
            raise NotImplementedError(f"Metric '{metric}' not implemented")
    else:
        # Pure Python version
        m = len(xa)
        n = len(xb)

        if metric == "euclidean":
            result = []
            for i in range(m):
                row = []
                for j in range(n):
                    if hasattr(xa[i], "__iter__"):
                        diff_sq = sum((a - b) ** 2 for a, b in zip(xa[i], xb[j]))
                    else:
                        diff_sq = (xa[i] - xb[j]) ** 2
                    row.append(math.sqrt(diff_sq))
                result.append(row)
            return np.array(result) if HAS_NUMPY else result
        else:
            raise NotImplementedError(f"Metric '{metric}' not implemented")


# === Special Functions ===

def logsumexp(a, axis=None, keepdims=False):
    """
    Compute log of sum of exponentials in a numerically stable way.

    log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))
    """
    if HAS_NUMPY:
        a = np.asarray(a)

        if axis is None:
            a_max = np.max(a)
            result = a_max + math.log(np.sum(np.exp(a - a_max)))
        else:
            a_max = np.max(a, axis=axis, keepdims=True)
            tmp = np.exp(a - a_max)
            s = np.sum(tmp, axis=axis, keepdims=keepdims)
            result = np.squeeze(a_max) + np.log(s)

        return result
    else:
        # Pure Python version
        if hasattr(a, "_get_flat"):
            flat = a._get_flat()
        elif hasattr(a, "__iter__"):
            flat = list(a)
        else:
            return float(a)

        a_max = max(flat)
        return a_max + math.log(sum(math.exp(x - a_max) for x in flat))
