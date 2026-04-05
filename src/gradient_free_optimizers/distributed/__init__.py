"""Distributed evaluation backends for parallel objective function evaluation.

Use a distribution backend as a decorator on your objective function
to enable parallel batch evaluation during optimization::

    from gradient_free_optimizers.distributed import Multiprocessing

    @Multiprocessing(n_workers=4).distribute
    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

Sync backends (Multiprocessing, Joblib) evaluate entire batches at once.
Async backends (Ray) process results as they arrive, keeping all workers
busy at all times.

Custom backends can be created by subclassing BaseDistribution and
implementing the _distribute method (and optionally _submit/_wait_any
for async support).
"""

from ._base import BaseDistribution
from ._dask import Dask
from ._joblib import Joblib
from ._multiprocessing import Multiprocessing
from ._ray import Ray

__all__ = ["BaseDistribution", "Dask", "Joblib", "Multiprocessing", "Ray"]
