========================
Distributed Evaluation
========================

When your objective function is expensive (model training, simulation, data
processing), you can evaluate multiple candidate positions in parallel across
separate worker processes. GFO provides a decorator-based approach that requires
minimal changes to your existing code.


.. grid:: 1

   .. grid-item-card::
      :class-card: sd-border-primary gfo-compact

      .. code-block:: python

          @Joblib(n_workers=4).distribute
          def objective(para):
              ...

          opt.search(objective, n_iter=100)


How It Works
------------

The ``@distribute`` decorator wraps your objective function so that the
optimizer evaluates positions in batches of ``n_workers`` instead of one at
a time. Each batch is sent to a pool of worker processes that run the
evaluations simultaneously.

.. code-block:: text

    Serial:    ask -> eval -> tell -> ask -> eval -> tell -> ask -> eval -> tell
    Batch:     ask(4) -> [eval, eval, eval, eval] -> tell(4) -> ask(4) -> ...

The optimizer proposes ``n_workers`` positions per batch, the decorated function
evaluates them in parallel, and the results are fed back. The initialization
phase (where the optimizer evaluates starting positions) always runs serially.

Async backends like Ray and Dask go one step further. Instead of waiting for
an entire batch, they process results individually as workers complete and
immediately submit new work. This keeps all workers busy at all times:

.. code-block:: text

    Async:     submit(4) -> W1 done -> tell(1), submit(1) -> W3 done -> tell(1), submit(1) -> ...


Basic Usage
-----------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer
    from gradient_free_optimizers.distributed import Joblib

    @Joblib(n_workers=4).distribute
    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=100)

The decorator can also be applied without the ``@`` syntax, which is useful when
the function is defined elsewhere:

.. code-block:: python

    distributed_objective = Ray(n_workers=8).distribute(objective)
    opt.search(distributed_objective, n_iter=100)


Machine Learning Example
------------------------

Parallel hyperparameter tuning where each worker trains and evaluates a
model independently:

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_wine

    from gradient_free_optimizers import BayesianOptimizer
    from gradient_free_optimizers.distributed import Joblib

    data = load_wine()
    X, y = data.data, data.target

    @Joblib(n_workers=4).distribute
    def model(para):
        gbc = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
        )
        scores = cross_val_score(gbc, X, y, cv=3)
        return scores.mean()

    search_space = {
        "n_estimators": np.arange(20, 120, 1),
        "max_depth": np.arange(2, 12, 1),
    }

    opt = BayesianOptimizer(search_space)
    opt.search(model, n_iter=50)


Available Backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Backend
     - Async
     - Dependencies
     - Use case
   * - ``Multiprocessing``
     - No
     - stdlib only
     - Local parallelism via fork. No extra dependencies needed.
   * - ``Joblib``
     - No
     - joblib
     - Local parallelism with automatic fork/spawn handling. Included with scikit-learn.
   * - ``Ray``
     - Yes
     - ray
     - Local or cluster-wide parallelism. Handles serialization via cloudpickle.
   * - ``Dask``
     - Yes
     - dask[distributed]
     - Local or cluster-wide parallelism. Integrates with existing Dask infrastructure.

Sync backends (Multiprocessing, Joblib) evaluate a full batch and wait for all
results before proposing the next batch. Async backends (Ray, Dask) process
results individually as they arrive, keeping workers busy at all times.


Async Mode
----------

When using an async backend like Ray, most optimizers run in true async mode
where each completed evaluation immediately triggers a new proposal. Three
optimizers (Downhill Simplex, Powell's Method, DIRECT) use a batch-async
fallback where positions are submitted asynchronously but collected per batch.
This distinction is handled automatically.

.. code-block:: python

    from gradient_free_optimizers import ParticleSwarmOptimizer
    from gradient_free_optimizers.distributed import Ray

    @Ray(n_workers=8).distribute
    def expensive_simulation(para):
        # Each evaluation takes 10-60 seconds
        return run_simulation(para)

    opt = ParticleSwarmOptimizer(search_space, population=20)
    opt.search(expensive_simulation, n_iter=200)

True async is most beneficial when evaluation times vary widely. Slow evaluations
no longer block fast ones from being processed and replaced.


Error Handling
--------------

The ``catch`` parameter works with distributed evaluation. Exceptions are caught
inside each worker process, and the fallback score is returned in place of the
failed evaluation:

.. code-block:: python

    @Joblib(n_workers=4).distribute
    def flaky_model(para):
        # Might fail for certain hyperparameter combinations
        return train_and_evaluate(para)

    opt.search(flaky_model, n_iter=100, catch={ValueError: -1000.0})


Custom Backends
---------------

Create your own distribution backend by subclassing ``BaseDistribution``
and implementing ``_distribute``:

.. code-block:: python

    from gradient_free_optimizers.distributed import BaseDistribution

    class MyClusterBackend(BaseDistribution):
        def _distribute(self, func, params_batch):
            scores = my_cluster.map(func, params_batch)
            return scores

    @MyClusterBackend(n_workers=16).distribute
    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

The ``_distribute`` method receives the original objective function and a
list of parameter dictionaries. It must return a list of scores in the
same order.

For async support, also implement ``_submit`` and ``_wait_any`` and set
``_is_async = True``:

.. code-block:: python

    class MyAsyncBackend(BaseDistribution):
        _is_async = True

        def _distribute(self, func, params_batch):
            futures = [self._submit(func, p) for p in params_batch]
            return [f.result() for f in futures]

        def _submit(self, func, params):
            return my_cluster.submit(func, params)

        def _wait_any(self, futures):
            done = my_cluster.wait_any(futures)
            return done, done.result()


Batch Size and Algorithm Interaction
------------------------------------

The batch size (``n_workers``) is independent of algorithm-internal
parameters like ``n_neighbours`` or ``population``. The optimizer produces
positions using its own logic, and the batch size only controls how many
are evaluated simultaneously:

.. code-block:: python

    # 20-particle PSO evaluated 4 at a time = 5 rounds per generation
    opt = ParticleSwarmOptimizer(search_space, population=20)
    opt.search(objective, n_iter=100)  # objective has n_workers=4

    # Hill Climbing with n_neighbours=5, evaluated 8 at a time
    opt = HillClimbingOptimizer(search_space, n_neighbours=5)
    opt.search(objective, n_iter=100)  # objective has n_workers=8

For surrogate model-based optimizers (Bayesian Optimization, TPE, Forest
Optimizer), batch positions are selected using KMeans clustering on the
acquisition landscape to ensure diversity. Without this, all batch positions
would cluster around the single highest acquisition peak.


Limitations
-----------

**The objective function** must be defined at module level (not a lambda
or closure) for the ``Multiprocessing`` backend on systems that do not
support the ``fork`` start method. Ray and Dask handle closures via
cloudpickle.
