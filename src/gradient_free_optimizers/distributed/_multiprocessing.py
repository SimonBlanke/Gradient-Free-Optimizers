"""Multiprocessing backend for distributed evaluation.

Uses a module-level variable to pass the objective function to worker
processes, avoiding pickling issues. Only the parameter dicts (plain
dicts of floats/strings) are serialized over the pool's pipe.
"""

from __future__ import annotations

import multiprocessing

from ._base import BaseDistribution

# Shared between parent and forked workers via copy-on-write memory.
# Set by _distribute() before pool.map, cleared after.
_worker_func = None


def _eval_single(params):
    """Evaluate a single parameter dict in a worker process."""
    return _worker_func(params)


class Multiprocessing(BaseDistribution):
    """Distribute evaluations across local processes via multiprocessing.Pool.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes. Use -1 for all available CPUs.

    Examples
    --------
    ::

        from gradient_free_optimizers.distributed import Multiprocessing

        @Multiprocessing(n_workers=4).distribute
        def objective(para):
            return -(para["x"]**2 + para["y"]**2)

        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=100)

    Notes
    -----
    Uses the ``fork`` multiprocessing context on Linux/macOS so that
    the objective function does not need to be picklable. On systems
    where ``fork`` is unavailable the function must be defined at module
    level (not a lambda or closure).
    """

    def __init__(self, n_workers: int = -1):
        if n_workers == -1:
            import os

            n_workers = os.cpu_count() or 1
        super().__init__(n_workers)

    def _distribute(self, func, params_batch):
        """Evaluate objective in parallel using multiprocessing.Pool.

        The function is stored in a module-level variable and inherited
        by forked workers, so only the params dicts travel through the
        serialization pipe.
        """
        global _worker_func
        _worker_func = func
        try:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(self.n_workers) as pool:
                scores = pool.map(_eval_single, params_batch)
        finally:
            _worker_func = None
        return scores
