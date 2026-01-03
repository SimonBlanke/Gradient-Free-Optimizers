"""Shared fixtures and helpers for array backend tests."""

import pytest

from gradient_free_optimizers._array_backend import _pure as pure_backend
from gradient_free_optimizers._array_backend import HAS_NUMPY

# Only import numpy backend if numpy is available
if HAS_NUMPY:
    from gradient_free_optimizers._array_backend import _numpy as np_backend
else:
    np_backend = None


def arrays_close(a, b, rtol=1e-5, atol=1e-8):
    """Check if two array-like objects are approximately equal."""
    a_list = to_list(a)
    b_list = to_list(b)

    if len(a_list) != len(b_list):
        return False

    for x, y in zip(a_list, b_list):
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


def to_list(arr):
    """Convert array-like to list for comparison."""
    if hasattr(arr, "_get_flat"):
        return arr._get_flat()
    if hasattr(arr, "tolist"):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return [arr]


@pytest.fixture
def sample_array():
    """Sample array for testing."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def sample_2d_array():
    """Sample 2D array for testing."""
    return [[1, 2, 3], [4, 5, 6]]
