"""Joblib backend for local parallel evaluation.

Uses joblib.Parallel for process- or thread-based parallelism.
Handles fork/spawn complexity and loky backend selection
automatically. Requires joblib to be installed (included with
scikit-learn).
"""

from __future__ import annotations

from ._base import BaseDistribution


class Joblib(BaseDistribution):
    """Distribute evaluations across local cores via joblib.

    A synchronous backend that uses joblib's ``Parallel`` for local
    multicore evaluation. Compared to :class:`Multiprocessing`, this
    backend handles fork-vs-spawn selection automatically and supports
    the ``loky`` backend for better process reuse.

    Parameters
    ----------
    n_workers : int, default=-1
        Number of parallel jobs. Use -1 for all available CPUs (default).
        Uses joblib's ``cpu_count()`` for detection.
    backend : str, default="loky"
        Joblib parallel backend. Common choices: ``"loky"`` (default,
        reusable process pool), ``"multiprocessing"`` (fresh processes),
        ``"threading"`` (threads, useful for GIL-free code like numpy).

    Examples
    --------
    ::

        from gradient_free_optimizers.distributed import Joblib

        @Joblib(n_workers=4).distribute
        def objective(para):
            return -(para["x"]**2 + para["y"]**2)

        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=100)
    """

    def __init__(self, n_workers: int = -1, backend: str = "loky"):
        from joblib import cpu_count

        if n_workers == -1:
            n_workers = cpu_count()
        super().__init__(n_workers)
        self._backend_name = backend

    def _distribute(self, func, params_batch):
        from joblib import Parallel, delayed

        return Parallel(n_jobs=self.n_workers, backend=self._backend_name)(
            delayed(func)(params) for params in params_batch
        )
