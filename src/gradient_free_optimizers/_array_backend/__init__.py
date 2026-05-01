"""
Array backend abstraction for optional NumPy dependency.

This module provides a unified interface for array operations with automatic
fallback: numpy (fastest) -> C extension (fast) -> pure Python (functional).

Usage:
    from gradient_free_optimizers._array_backend import array, zeros, clip, rint
"""

try:
    import numpy

    _ = numpy.__version__
    from numpy import array as _test_array

    del _test_array
    HAS_NUMPY = True
except (ImportError, AttributeError):
    HAS_NUMPY = False

try:
    from . import _fast_ops  # noqa: F401

    HAS_C_EXTENSION = True
except ImportError:
    HAS_C_EXTENSION = False

if HAS_NUMPY:
    from ._numpy import *

    _backend_name = "numpy"
    ndarray = numpy.ndarray
elif HAS_C_EXTENSION:
    from ._c_extension import *
    from ._pure import GFOArray

    _backend_name = "c_extension"
    ndarray = GFOArray
else:
    from ._pure import *
    from ._pure import GFOArray

    _backend_name = "pure"
    ndarray = GFOArray


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
    "triu",
    "invert",
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
    "ndarray",
]
