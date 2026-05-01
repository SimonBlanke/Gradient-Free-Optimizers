"""Pure Python math backend optimized for performance.

Provides scipy-compatible math functions using only the Python stdlib
and the GFO array backend. When numpy is available (via the array backend),
vectorized fast-paths are used for norm_cdf, norm_pdf, cdist, and logsumexp.
"""

import math

from gradient_free_optimizers._array_backend import HAS_NUMPY
from gradient_free_optimizers._array_backend import array as arr_array

if HAS_NUMPY:
    import numpy as np

_sqrt = math.sqrt
_exp = math.exp
_log = math.log
_erf = math.erf
_dist = math.dist
_fabs = math.fabs

_SQRT2 = _sqrt(2.0)
_INV_SQRT2 = 1.0 / _SQRT2
_SQRT2PI = _sqrt(2.0 * math.pi)


def _to_nested_lists(m):
    """Convert a 2D array-like to list-of-lists of Python floats."""
    if hasattr(m, "tolist"):
        return m.tolist()
    return [list(row) for row in m]


def _to_flat_list(v):
    """Convert a 1D array-like to a flat Python list."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "_data"):
        return list(v._data)
    return list(v)


def _numpy_erf(x):
    """Vectorized erf via Abramowitz & Stegun approximation in numpy."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-(x * x))
    return sign * y


def norm_cdf(x, loc=0, scale=1):
    """Cumulative distribution function of normal distribution."""
    if HAS_NUMPY and hasattr(x, "__iter__"):
        x = np.asarray(x, dtype=float)
        z = (x - loc) / scale
        return 0.5 * (1.0 + _numpy_erf(z * _INV_SQRT2))

    if hasattr(x, "_data"):
        inv_scale = 1.0 / scale
        shape = x._shape if hasattr(x, "_shape") else (len(x),)
        result = arr_array(
            [0.5 * (1.0 + _erf((xi - loc) * inv_scale * _INV_SQRT2)) for xi in x._data]
        )
        return result.reshape(shape) if len(shape) > 1 else result

    if hasattr(x, "__iter__"):
        inv_scale = 1.0 / scale
        return arr_array(
            [0.5 * (1.0 + _erf((xi - loc) * inv_scale * _INV_SQRT2)) for xi in x]
        )

    z = (x - loc) / scale
    return 0.5 * (1.0 + _erf(z * _INV_SQRT2))


def norm_pdf(x, loc=0, scale=1):
    """Probability density function of normal distribution."""
    if HAS_NUMPY and hasattr(x, "__iter__"):
        x = np.asarray(x, dtype=float)
        z = (x - loc) / scale
        return np.exp(-0.5 * z * z) / (scale * _SQRT2PI)

    if hasattr(x, "_data"):
        inv_scale = 1.0 / scale
        coeff = inv_scale / _SQRT2PI
        shape = x._shape if hasattr(x, "_shape") else (len(x),)
        result = arr_array(
            [coeff * _exp(-0.5 * ((xi - loc) * inv_scale) ** 2) for xi in x._data]
        )
        return result.reshape(shape) if len(shape) > 1 else result

    if hasattr(x, "__iter__"):
        inv_scale = 1.0 / scale
        coeff = inv_scale / _SQRT2PI
        return arr_array(
            [coeff * _exp(-0.5 * ((xi - loc) * inv_scale) ** 2) for xi in x]
        )

    z = (x - loc) / scale
    return _exp(-0.5 * z * z) / (scale * _SQRT2PI)


def cholesky(a, lower=True):
    """Cholesky decomposition of a positive-definite matrix.

    Returns lower triangular L such that A = L @ L.T.
    """
    rows = _to_nested_lists(a)
    n = len(rows)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L_i = L[i]
        for j in range(i + 1):
            L_j = L[j]
            s = 0.0
            for k in range(j):
                s += L_i[k] * L_j[k]

            if i == j:
                val = rows[i][i] - s
                if val < 0:
                    raise ValueError("Matrix is not positive definite")
                L_i[j] = _sqrt(val)
            else:
                denom = L_j[j]
                L_i[j] = (rows[i][j] - s) / denom if denom != 0 else 0.0

    if not lower:
        L = [[L[j][i] for j in range(n)] for i in range(n)]

    return arr_array(L)


def cho_solve(c_and_lower, b):
    """Solve A @ x = b given Cholesky factor L where A = L @ L.T."""
    L, lower = c_and_lower
    L_data = _to_nested_lists(L)
    b_data = _to_flat_list(b)
    n = len(L_data)

    y = [0.0] * n
    for i in range(n):
        s = b_data[i]
        L_i = L_data[i]
        for j in range(i):
            s -= L_i[j] * y[j]
        y[i] = s / L_i[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L_data[j][i] * x[j]
        x[i] = s / L_data[i][i]

    return arr_array(x)


def solve(a, b, assume_a=None):
    """Solve a @ x = b via Gaussian elimination with partial pivoting."""
    a_data = [list(row) for row in _to_nested_lists(a)]
    b_data = list(_to_flat_list(b))
    n = len(a_data)

    for i in range(n):
        max_row = i
        max_val = _fabs(a_data[i][i])
        for k in range(i + 1, n):
            v = _fabs(a_data[k][i])
            if v > max_val:
                max_val = v
                max_row = k

        if max_row != i:
            a_data[i], a_data[max_row] = a_data[max_row], a_data[i]
            b_data[i], b_data[max_row] = b_data[max_row], b_data[i]

        pivot = a_data[i][i]
        if pivot == 0:
            continue

        a_i = a_data[i]
        for k in range(i + 1, n):
            a_k = a_data[k]
            c = a_k[i] / pivot
            for j in range(i, n):
                a_k[j] -= c * a_i[j]
            b_data[k] -= c * b_data[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b_data[i]
        a_i = a_data[i]
        for j in range(i + 1, n):
            s -= a_i[j] * x[j]
        if a_i[i] != 0:
            x[i] = s / a_i[i]

    return arr_array(x)


def _solve_triangular_1d(a_data, b_list, lower):
    """Solve triangular system for a single right-hand side."""
    n = len(b_list)
    x = [0.0] * n

    if lower:
        for i in range(n):
            s = b_list[i]
            a_i = a_data[i]
            for j in range(i):
                s -= a_i[j] * x[j]
            x[i] = s / a_i[i]
    else:
        for i in range(n - 1, -1, -1):
            s = b_list[i]
            a_i = a_data[i]
            for j in range(i + 1, n):
                s -= a_i[j] * x[j]
            x[i] = s / a_i[i]

    return x


def solve_triangular(a, b, lower=True):
    """Solve triangular linear system a @ x = b."""
    a_data = _to_nested_lists(a)

    if hasattr(b, "tolist"):
        b_raw = b.tolist()
    elif hasattr(b, "_data"):
        if hasattr(b, "ndim") and b.ndim == 2:
            b_raw = b.tolist()
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
    if xa_data and not isinstance(xa_data[0], list):
        xa_data = [[v] for v in xa_data]
    if xb_data and not isinstance(xb_data[0], list):
        xb_data = [[v] for v in xb_data]

    m = len(xa_data)
    n = len(xb_data)
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row_a = xa_data[i]
        result_i = result[i]
        for j in range(n):
            result_i[j] = _dist(row_a, xb_data[j])

    return arr_array(result)


def logsumexp(a, axis=None, keepdims=False):
    """Compute log(sum(exp(a))) in a numerically stable way."""
    if HAS_NUMPY:
        a = np.asarray(a)

        if axis is None:
            a_max = np.max(a)
            result = a_max + _log(float(np.sum(np.exp(a - a_max))))
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
    total = 0.0
    for x in flat:
        total += _exp(x - a_max)
    return a_max + _log(total)
