# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Normalization utilities for SMBO optimizers."""

import numpy as np


def normalize(array):
    """Normalize array values to [0, 1] range using min-max scaling.

    Parameters
    ----------
    array : np.ndarray
        Input array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1].
        If all values are identical (range=0), returns random values
        to avoid division by zero and enable exploration.
    """
    array_min = array.min()
    array_max = array.max()
    range_ = array_max - array_min

    if range_ == 0:
        return np.random.random_sample(array.shape)
    else:
        return (array - array_min) / range_
