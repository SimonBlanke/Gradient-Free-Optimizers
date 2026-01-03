"""
SciPy backend - thin wrapper around SciPy for unified interface.

This module re-exports SciPy functions used by GFO, providing a consistent
interface that can be swapped with the pure Python backend.
"""

from scipy import stats
from scipy import linalg
from scipy import optimize
from scipy.spatial import distance
from scipy import special

# === Statistical Functions ===

def norm_cdf(x, loc=0, scale=1):
    """Cumulative distribution function of normal distribution."""
    return stats.norm.cdf(x, loc=loc, scale=scale)

def norm_pdf(x, loc=0, scale=1):
    """Probability density function of normal distribution."""
    return stats.norm.pdf(x, loc=loc, scale=scale)

# === Linear Algebra ===

def cholesky(a, lower=True):
    """Cholesky decomposition."""
    return linalg.cholesky(a, lower=lower)

def cho_solve(c_and_lower, b):
    """Solve using Cholesky decomposition."""
    return linalg.cho_solve(c_and_lower, b)

def solve(a, b):
    """Solve linear system."""
    return linalg.solve(a, b)

# === Optimization ===

def minimize(fun, x0, method=None, bounds=None, options=None):
    """Minimize a function."""
    return optimize.minimize(fun, x0, method=method, bounds=bounds, options=options)

# === Distance Functions ===

def cdist(xa, xb, metric="euclidean"):
    """Compute pairwise distances."""
    return distance.cdist(xa, xb, metric=metric)

# === Special Functions ===

def logsumexp(a, axis=None, keepdims=False):
    """Log of sum of exponentials."""
    return special.logsumexp(a, axis=axis, keepdims=keepdims)
