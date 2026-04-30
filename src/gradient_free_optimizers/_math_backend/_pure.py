"""
Pure Python math functions - SciPy-compatible interface without dependencies.

This module provides pure Python implementations of mathematical functions
used by GFO that would normally come from SciPy. When NumPy is available,
it is used for performance-critical operations (cdist, logsumexp) but is
not required for correctness.
"""

import math

from gradient_free_optimizers._array_backend import HAS_NUMPY
from gradient_free_optimizers._array_backend import array as arr_array

if HAS_NUMPY:
    import numpy as np


def _to_nested_lists(m):
    """Extract a 2D array-like as list-of-lists of Python numbers."""
    if hasattr(m, "tolist"):
        return m.tolist()
    if hasattr(m, "_data") and hasattr(m, "ndim") and m.ndim == 2:
        return [list(row) for row in m._data]
    return [list(row) for row in m]


def _to_flat_list(v):
    """Extract a 1D array-like as a flat Python list."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "_data"):
        return list(v._data)
    return list(v)


def _numpy_erf(x):
    """Vectorized erf via Abramowitz & Stegun, runs entirely in numpy C."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        (
            (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592
        )
        * t
        * np.exp(-x * x)
    )
    return sign * y


def norm_cdf(x, loc=0, scale=1):
    """Cumulative distribution function of normal distribution."""
    if HAS_NUMPY and hasattr(x, "__iter__"):
        x = np.asarray(x, dtype=float)
        z = (x - loc) / scale
        return 0.5 * (1.0 + _numpy_erf(z / math.sqrt(2)))

    if hasattr(x, "__iter__"):
        return arr_array([norm_cdf(xi, loc, scale) for xi in x])

    z = (x - loc) / scale
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def norm_pdf(x, loc=0, scale=1):
    """Probability density function of normal distribution."""
    if HAS_NUMPY and hasattr(x, "__iter__"):
        x = np.asarray(x, dtype=float)
        z = (x - loc) / scale
        return np.exp(-0.5 * z * z) / (scale * math.sqrt(2 * math.pi))

    if hasattr(x, "__iter__"):
        return arr_array([norm_pdf(xi, loc, scale) for xi in x])

    z = (x - loc) / scale
    return math.exp(-0.5 * z * z) / (scale * math.sqrt(2 * math.pi))


def cholesky(a, lower=True):
    """
    Cholesky decomposition of a positive-definite matrix.

    Returns lower triangular matrix L such that A = L @ L.T
    """
    a_data = _to_nested_lists(a)
    n = len(a_data)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = math.fsum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = a_data[i][i] - s
                if val < 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    L[i][j] = 0
                else:
                    L[i][j] = (a_data[i][j] - s) / L[j][j]

    if not lower:
        L = [[L[j][i] for j in range(n)] for i in range(n)]

    return arr_array(L)


def cho_solve(c_and_lower, b):
    """
    Solve A @ x = b given Cholesky factor L where A = L @ L.T.

    Uses forward then backward substitution.
    """
    L, lower = c_and_lower
    L_data = _to_nested_lists(L)
    b_data = _to_flat_list(b)
    n = len(L_data)

    y = [0.0] * n
    for i in range(n):
        s = math.fsum(L_data[i][j] * y[j] for j in range(i))
        y[i] = (b_data[i] - s) / L_data[i][i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum(L_data[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / L_data[i][i]

    return arr_array(x)


def solve(a, b, assume_a=None):
    """Solve a @ x = b via Gaussian elimination with partial pivoting."""
    a_data = [list(row) for row in _to_nested_lists(a)]
    b_data = list(_to_flat_list(b))
    n = len(a_data)

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(a_data[k][i]) > abs(a_data[max_row][i]):
                max_row = k

        a_data[i], a_data[max_row] = a_data[max_row], a_data[i]
        b_data[i], b_data[max_row] = b_data[max_row], b_data[i]

        for k in range(i + 1, n):
            if a_data[i][i] != 0:
                c = a_data[k][i] / a_data[i][i]
                for j in range(i, n):
                    a_data[k][j] -= c * a_data[i][j]
                b_data[k] -= c * b_data[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b_data[i]
        for j in range(i + 1, n):
            x[i] -= a_data[i][j] * x[j]
        if a_data[i][i] != 0:
            x[i] /= a_data[i][i]

    return arr_array(x)


def _solve_triangular_1d(a_data, b_list, lower):
    """Solve triangular system for a single right-hand side."""
    n = len(b_list)
    x = [0.0] * n

    if lower:
        for i in range(n):
            s = b_list[i]
            for j in range(i):
                s -= a_data[i][j] * x[j]
            x[i] = s / a_data[i][i]
    else:
        for i in range(n - 1, -1, -1):
            s = b_list[i]
            for j in range(i + 1, n):
                s -= a_data[i][j] * x[j]
            x[i] = s / a_data[i][i]

    return x


def solve_triangular(a, b, lower=True):
    """Solve triangular linear system a @ x = b."""
    a_data = _to_nested_lists(a)

    if hasattr(b, "tolist"):
        b_raw = b.tolist()
    elif hasattr(b, "_data"):
        if hasattr(b, "ndim") and b.ndim == 2:
            b_raw = [list(row) for row in b._data]
        else:
            b_raw = list(b._data)
    else:
        b_raw = list(b)

    is_2d = isinstance(b_raw, list) and b_raw and isinstance(b_raw[0], list)

    if not is_2d:
        return arr_array(_solve_triangular_1d(a_data, list(b_raw), lower))

    n_rows = len(b_raw)
    n_cols = len(b_raw[0])
    result = [[0.0] * n_cols for _ in range(n_rows)]
    for j in range(n_cols):
        col = [b_raw[i][j] for i in range(n_rows)]
        x_col = _solve_triangular_1d(a_data, col, lower)
        for i in range(n_rows):
            result[i][j] = x_col[i]
    return arr_array(result)


class OptimizeResult:
    """Simple result container matching scipy.optimize.OptimizeResult."""

    def __init__(self, x, fun, success, message=""):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message


def minimize(fun, x0, method=None, bounds=None, options=None):
    """Minimize a function using coordinate descent with shrinking step size."""
    x = list(_to_flat_list(x0))
    options = options or {}
    maxiter = options.get("maxiter", 100)
    tol = options.get("gtol", 1e-5)

    best_f = fun(x)
    step = 0.1

    for _iteration in range(maxiter):
        improved = False

        for i in range(len(x)):
            x_new = list(x)
            x_new[i] += step

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

            x_new = list(x)
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
        message="Optimization terminated successfully.",
    )


def cdist(xa, xb, metric="euclidean"):
    """Compute pairwise Euclidean distances between two sets of points."""
    if metric != "euclidean":
        raise NotImplementedError(f"Metric '{metric}' not implemented")

    if HAS_NUMPY:
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)
        if xa.ndim == 1:
            xa = xa.reshape(-1, 1)
        if xb.ndim == 1:
            xb = xb.reshape(-1, 1)
        diff = xa[:, None, :] - xb[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    xa_data = _to_nested_lists(xa)
    xb_data = _to_nested_lists(xb)
    if xa_data and not isinstance(xa_data[0], list | tuple):
        xa_data = [[v] for v in xa_data]
    if xb_data and not isinstance(xb_data[0], list | tuple):
        xb_data = [[v] for v in xb_data]

    m = len(xa_data)
    n = len(xb_data)
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            sq_dist = math.fsum((a - b) ** 2 for a, b in zip(xa_data[i], xb_data[j]))
            result[i][j] = math.sqrt(sq_dist)

    return arr_array(result)


def logsumexp(a, axis=None, keepdims=False):
    """Compute log(sum(exp(a))) in a numerically stable way."""
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

    if hasattr(a, "_get_flat"):
        flat = a._get_flat()
    elif hasattr(a, "__iter__"):
        flat = list(a)
    else:
        return float(a)

    a_max = max(flat)
    return a_max + math.log(math.fsum(math.exp(x - a_max) for x in flat))
