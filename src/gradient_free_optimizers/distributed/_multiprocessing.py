"""Multiprocessing backend for distributed evaluation.

On platforms with ``fork`` (Linux, older macOS), uses a module-level
variable to pass the objective function to workers via shared memory,
avoiding pickling entirely. On ``spawn`` platforms (Windows, macOS
3.14+), the function is pickled alongside each parameter dict using
``starmap``, which requires it to be importable at module level.
"""

from __future__ import annotations

import multiprocessing

from ._base import BaseDistribution

# Shared between parent and forked workers via copy-on-write memory.
# Only used with the fork context.
_worker_func = None


def _eval_single(params):
    """Worker entry point for fork context."""
    return _worker_func(params)


def _eval_with_func(func, params):
    """Worker entry point for spawn context (func is pickled per call)."""
    return func(params)


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
    Prefers the ``fork`` context (Linux/macOS) so the objective function
    is inherited by workers without pickling. Falls back to the platform
    default (``spawn`` on Windows, macOS 3.14+) where ``fork`` is
    unavailable. With ``spawn``, the objective must be defined at module
    level (not a lambda or closure).
    """

    def __init__(self, n_workers: int = -1):
        if n_workers == -1:
            import os

            n_workers = os.cpu_count() or 1
        super().__init__(n_workers)
        self._mp_context = self._select_context()
        self._use_fork = self._mp_context.get_start_method() == "fork"

    @staticmethod
    def _select_context():
        available = multiprocessing.get_all_start_methods()
        if "fork" in available:
            return multiprocessing.get_context("fork")
        return multiprocessing.get_context()

    def _distribute(self, func, params_batch):
        global _worker_func

        if self._use_fork:
            # Set before Pool creation so forked workers inherit it
            _worker_func = func
            with self._mp_context.Pool(self.n_workers) as pool:
                results = pool.map(_eval_single, params_batch)
            _worker_func = None
        else:
            with self._mp_context.Pool(self.n_workers) as pool:
                results = pool.starmap(
                    _eval_with_func,
                    [(func, p) for p in params_batch],
                )
        return results
