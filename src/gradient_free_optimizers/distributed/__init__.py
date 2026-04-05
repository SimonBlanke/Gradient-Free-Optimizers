"""Distributed evaluation backends for parallel objective function evaluation.

Use a distribution backend as a decorator on your objective function
to enable parallel batch evaluation during optimization::

    from gradient_free_optimizers.distributed import Multiprocessing

    @Multiprocessing(n_workers=4).distribute
    def objective(x, y):
        return -(x**2 + y**2)

Custom backends can be created by subclassing BaseDistribution and
implementing the _distribute method.
"""

from ._base import BaseDistribution
from ._multiprocessing import Multiprocessing

__all__ = ["BaseDistribution", "Multiprocessing"]
