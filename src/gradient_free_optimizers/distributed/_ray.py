"""Ray backend for distributed evaluation.

Supports both synchronous batch evaluation and asynchronous
per-result processing. Ray handles serialization via cloudpickle,
so closures and lambda functions work without special handling.

Requires ray to be installed (``pip install ray``).
"""

from __future__ import annotations

from ._base import BaseDistribution


class Ray(BaseDistribution):
    """Distribute evaluations via Ray, locally or across a cluster.

    An async-capable backend that uses Ray for parallel evaluation.
    When used with an async-compatible optimizer, results are processed
    individually as workers complete, keeping all workers busy at all
    times. For stateful optimizers (Simplex, Powell's, DIRECT), the
    search loop automatically falls back to batch-async mode where
    positions are submitted asynchronously but results are collected
    per batch.

    Ray initializes lazily on first use. If Ray is already initialized
    (e.g., connected to an existing cluster), that session is reused.

    Parameters
    ----------
    n_workers : int, default=1
        Number of concurrent evaluation tasks. Unlike Multiprocessing
        or Joblib, this does not map to CPU cores directly. Ray
        schedules tasks across available resources.

    Examples
    --------
    Local usage::

        from gradient_free_optimizers.distributed import Ray

        @Ray(n_workers=4).distribute
        def objective(para):
            return -(para["x"]**2 + para["y"]**2)

        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=100)

    With an existing Ray cluster::

        import ray
        ray.init(address="auto")

        @Ray(n_workers=16).distribute
        def objective(para):
            return expensive_simulation(para)
    """

    _is_async = True

    def __init__(self, n_workers: int = 1):
        super().__init__(n_workers)
        self._remote_cache = {}

    def _ensure_init(self):
        """Initialize Ray if not already running."""
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _remote(self, func):
        """Get or create a ray.remote wrapper for a function.

        Caches the remote wrapper by function id to avoid repeated
        serialization of the same function.
        """
        import ray

        self._ensure_init()
        key = id(func)
        cached = self._remote_cache.get(key)
        if cached is not None:
            return cached
        remote = ray.remote(func)
        self._remote_cache[key] = remote
        return remote

    def _distribute(self, func, params_batch):
        import ray

        remote = self._remote(func)
        futures = [remote.remote(p) for p in params_batch]
        return ray.get(futures)

    def _submit(self, func, params):
        remote = self._remote(func)
        return remote.remote(params)

    def _wait_any(self, futures):
        import ray

        future_list = list(futures)
        done, _ = ray.wait(future_list, num_returns=1)
        completed = done[0]
        result = ray.get(completed)
        return completed, result
