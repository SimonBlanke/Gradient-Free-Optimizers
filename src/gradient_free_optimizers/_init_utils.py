# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Utility functions for creating default initialization values.

These functions exist to avoid mutable default arguments in function signatures,
which is a common Python pitfall where dict/list defaults are shared across
all instances.
"""

from __future__ import annotations


def get_default_initialize() -> dict[str, int]:
    """Return a fresh copy of the default initialization dict.

    Returns
    -------
    dict[str, int]
        Default initialization strategy with keys:
        - "grid": Number of positions from a grid pattern
        - "random": Number of random positions
        - "vertices": Number of corner positions
    """
    return {"grid": 4, "random": 2, "vertices": 4}


def get_default_sampling() -> dict[str, int]:
    """Return a fresh copy of the default sampling dict.

    Returns
    -------
    dict[str, int]
        Default sampling strategy for SMBO methods.
    """
    return {"random": 1000000}
