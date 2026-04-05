===================
Evaluation Storage
===================

By default, GFO caches objective function evaluations in an in-memory
dictionary (``memory=True``). The storage system extends this with pluggable
backends that control where and how evaluation results are persisted.


.. grid:: 1

   .. grid-item-card::
      :class-card: sd-border-primary gfo-compact

      .. code-block:: python

          storage = SQLiteStorage("results.db")
          opt.search(objective, n_iter=100, memory=storage)


Why Use Storage
---------------

The default in-memory cache disappears when the Python process exits. A
persistent storage backend like ``SQLiteStorage`` keeps results on disk,
enabling two workflows that are otherwise impossible.

**Crash recovery.** If a long-running optimization is interrupted, restarting
with the same storage file skips all previously evaluated positions. The
optimizer begins proposing new positions immediately instead of re-evaluating
from scratch.

**Incremental optimization.** Run a short exploration first, inspect the
results, then continue with a different optimizer or more iterations. All
runs share the same evaluation cache as long as they point to the same
storage.


In-Memory Storage
-----------------

The default ``memory=True`` creates a ``MemoryStorage`` internally. You can
also create one explicitly if you need access to it after the search:

.. code-block:: python

    from gradient_free_optimizers.storage import MemoryStorage

    storage = MemoryStorage()
    opt.search(objective, n_iter=100, memory=storage)

    print(f"Cached {len(storage)} evaluations")


SQLite Storage
--------------

``SQLiteStorage`` writes evaluations to a local SQLite database. It uses WAL
journal mode for concurrent read access, making it safe for multiple processes
on the same machine.

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer
    from gradient_free_optimizers.storage import SQLiteStorage

    search_space = {"x": np.linspace(-10, 10, 1000)}
    storage = SQLiteStorage("optimization.db")

    # First run
    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=500, memory=storage)

    # Later: resume from cached results
    opt2 = HillClimbingOptimizer(search_space)
    opt2.search(objective, n_iter=500, memory=storage)

The second search skips positions that were already evaluated in the first run.
Both score and metrics are persisted.

SQLiteStorage is designed for single-machine use. It should not be placed on
network filesystems (NFS, SMB) where SQLite's file locking is unreliable.


Distributed + Storage
---------------------

Storage backends work with :doc:`distributed evaluation <distributed>`.
Before dispatching positions to workers, the search loop checks the cache
and only sends uncached positions. This reduces redundant work across
batches and enables crash recovery for distributed runs:

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer
    from gradient_free_optimizers.distributed import Joblib
    from gradient_free_optimizers.storage import SQLiteStorage

    @Joblib(n_workers=4).distribute
    def model(para):
        return expensive_training(para)

    storage = SQLiteStorage("distributed_results.db")
    opt = BayesianOptimizer(search_space)
    opt.search(model, n_iter=100, memory=storage)


Custom Backends
---------------

Create your own storage backend by subclassing ``BaseStorage`` and
implementing ``get``, ``put``, and ``contains``:

.. code-block:: python

    from gradient_free_optimizers.storage import BaseStorage

    class RedisStorage(BaseStorage):
        def __init__(self, url):
            self._client = redis.Redis.from_url(url)

        def get(self, key):
            data = self._client.get(str(key))
            if data is None:
                return None
            score, metrics = json.loads(data)
            return Result(score, metrics)

        def put(self, key, result):
            data = json.dumps([result.score, result.metrics])
            self._client.set(str(key), data)

        def contains(self, key):
            return self._client.exists(str(key))

Pass the instance to ``memory``:

.. code-block:: python

    storage = RedisStorage("redis://localhost:6379")
    opt.search(objective, n_iter=100, memory=storage)
