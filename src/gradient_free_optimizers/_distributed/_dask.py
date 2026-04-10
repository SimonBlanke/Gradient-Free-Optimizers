"""Dask backend for distributed evaluation.

Supports both synchronous batch evaluation and asynchronous
per-result processing via dask.distributed. Works with local
clusters (automatic) and remote schedulers.

Requires dask[distributed] (``pip install dask[distributed]``).
"""

from __future__ import annotations

from ._base import BaseDistribution


class Dask(BaseDistribution):
    """Distribute evaluations via Dask, locally or across a cluster.

    An async-capable backend that uses ``dask.distributed`` for parallel
    evaluation. Creates a local cluster automatically when no client or
    address is provided. Reuses an existing cluster when a ``Client``
    instance or scheduler address is passed.

    Parameters
    ----------
    n_workers : int, default=1
        Number of concurrent evaluation tasks. When creating a local
        cluster, this sets the number of worker processes.
    client : dask.distributed.Client or None, default=None
        An existing Dask client to reuse. Takes precedence over
        ``address``. The caller is responsible for managing its
        lifecycle.
    address : str or None, default=None
        Scheduler address to connect to (e.g. ``"tcp://scheduler:8786"``).
        Ignored if ``client`` is provided.

    Examples
    --------
    Local cluster (created automatically)::

        from gradient_free_optimizers._distributed import Dask

        @Dask(n_workers=4).distribute
        def objective(para):
            return -(para["x"]**2 + para["y"]**2)

    Existing cluster::

        from dask.distributed import Client

        client = Client("tcp://scheduler:8786")

        @Dask(n_workers=16, client=client).distribute
        def objective(para):
            return expensive_simulation(para)
    """

    _is_async = True

    def __init__(
        self,
        n_workers: int = 1,
        client=None,
        address: str | None = None,
    ):
        super().__init__(n_workers)
        self._client_arg = client
        self._address = address
        self._client = None

    def _get_client(self):
        """Get or create the Dask client (lazy initialization)."""
        if self._client is not None:
            return self._client

        from dask.distributed import Client

        if self._client_arg is not None:
            self._client = self._client_arg
        elif self._address is not None:
            self._client = Client(self._address)
        else:
            self._client = Client(
                n_workers=self.n_workers,
                threads_per_worker=1,
            )

        return self._client

    def _distribute(self, func, params_batch):
        client = self._get_client()
        futures = client.map(func, params_batch)
        return client.gather(futures)

    def _submit(self, func, params):
        client = self._get_client()
        return client.submit(func, params)

    def _wait_any(self, futures):
        from dask.distributed import wait

        result = wait(list(futures), return_when="FIRST_COMPLETED")
        completed = next(iter(result.done))
        return completed, completed.result()
