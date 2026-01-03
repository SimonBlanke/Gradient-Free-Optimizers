"""Shared fixtures and helpers for math backend tests."""

import pytest

from gradient_free_optimizers._math_backend import _pure as pure_math
from gradient_free_optimizers._math_backend import HAS_SCIPY

# Only import scipy backend if scipy is available
if HAS_SCIPY:
    from gradient_free_optimizers._math_backend import _scipy as scipy_backend
else:
    scipy_backend = None
