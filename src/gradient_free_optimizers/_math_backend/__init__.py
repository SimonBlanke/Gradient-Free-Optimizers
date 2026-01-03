"""
Math backend abstraction for optional SciPy dependency.

This module provides a unified interface for mathematical functions
(statistics, linear algebra, distance calculations) with automatic fallback
to pure Python implementations when SciPy is not available.

Usage:
    from gradient_free_optimizers._math_backend import norm_cdf, norm_pdf, cdist

The backend automatically selects the fastest available implementation:
- If SciPy is installed: uses SciPy (fast, numerically stable)
- If not: uses pure Python implementations (slower but functional)
"""

# === Dependency Detection ===

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# === Backend Selection ===

if HAS_SCIPY:
    from ._scipy import *
    _backend_name = "scipy"
else:
    from ._pure import *
    _backend_name = "pure"


__all__ = [
    "HAS_SCIPY",
    "_backend_name",
    # Statistical functions
    "norm_cdf",
    "norm_pdf",
    # Linear algebra
    "cholesky",
    "cho_solve",
    "solve",
    "solve_triangular",
    # Optimization
    "minimize",
    # Distance functions
    "cdist",
    # Special functions
    "logsumexp",
]
