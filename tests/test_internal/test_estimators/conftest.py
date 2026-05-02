"""Shared fixtures for estimator tests."""

import pytest


@pytest.fixture
def regression_data():
    """Y = x0 + 2*x1, perfectly learnable by a decision tree."""
    X = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 2.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [0.5, 0.5],
    ]
    y = [0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 5.0, 4.0, 1.5]
    return X, y


@pytest.fixture
def step_data():
    """Step function: y=1 for x0 <= 1, y=5 for x0 >= 2."""
    X = [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.5, 0.5],
        [3.0, 1.0],
    ]
    y = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    return X, y


@pytest.fixture
def constant_data():
    """Constant target data."""
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    y = [5.0, 5.0, 5.0, 5.0]
    return X, y
