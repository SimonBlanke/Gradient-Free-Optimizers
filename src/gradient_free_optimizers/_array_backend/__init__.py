"""
Array backend abstraction for optional NumPy dependency.

This module provides a unified interface for array operations with automatic
fallback to pure Python implementations when NumPy is not available.

Usage:
    from gradient_free_optimizers._array_backend import array, zeros, clip, rint

The backend automatically selects the fastest available implementation:
- If NumPy is installed: uses NumPy (fast)
- If not: uses pure Python GFOArray (slower but functional)
"""

# === Dependency Detection ===

try:
    import numpy
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# === Backend Selection ===

if HAS_NUMPY:
    from ._numpy import *
    _backend_name = "numpy"
else:
    from ._pure import *
    _backend_name = "pure"


__all__ = [
    "HAS_NUMPY",
    "_backend_name",
    # Array creation
    "array",
    "asarray",
    "zeros",
    "zeros_like",
    "ones",
    "empty",
    "empty_like",
    "full",
    "arange",
    "linspace",
    "meshgrid",
    "eye",
    "diag",
    # Type conversion
    "int32",
    "int64",
    "float32",
    "float64",
    # Mathematical operations
    "sum",
    "mean",
    "std",
    "var",
    "prod",
    "cumsum",
    # Element-wise math
    "exp",
    "log",
    "log10",
    "sqrt",
    "abs",
    "power",
    "square",
    "sin",
    "cos",
    # Rounding and clipping
    "clip",
    "rint",
    "round",
    "floor",
    "ceil",
    # Comparison and logic
    "maximum",
    "minimum",
    "greater",
    "less",
    "equal",
    "isnan",
    "isinf",
    "isfinite",
    # Index operations
    "argmax",
    "argmin",
    "argsort",
    "where",
    "nonzero",
    "searchsorted",
    "take",
    # Set operations
    "unique",
    "intersect1d",
    "isin",
    # Array manipulation
    "reshape",
    "transpose",
    "ravel",
    "flatten",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "tile",
    "repeat",
    "array_split",
    "split",
    # Linear algebra
    "dot",
    "matmul",
    "outer",
    "linalg",
    # Random number generation
    "random",
    # Constants
    "inf",
    "pi",
    "e",
    "nan",
    # Utility
    "copy",
    "allclose",
    "all",
    "any",
    "ndim",
    "shape",
]
