# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Normalization utilities for SMBO optimizers."""

from __future__ import annotations

from gradient_free_optimizers._array_backend import random


def normalize(arr):
    """Normalize array values to [0, 1] range using min-max scaling.

    Parameters
    ----------
    arr : np.ndarray
        Input array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1].
        If all values are identical (range=0), returns random values
        to avoid division by zero and enable exploration.
    """
    array_min = arr.min()
    array_max = arr.max()
    range_ = array_max - array_min

    if range_ == 0:
        return random.random_sample(arr.shape)
    else:
        return (arr - array_min) / range_
