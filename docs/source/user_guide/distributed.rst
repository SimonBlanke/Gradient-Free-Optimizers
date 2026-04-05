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

          @Multiprocessing(n_workers=4).distribute
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


Basic Usage
-----------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer
    from gradient_free_optimizers.distributed import Multiprocessing

    @Multiprocessing(n_workers=4).distribute
    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=100)


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
    from gradient_free_optimizers.distributed import Multiprocessing

    data = load_wine()
    X, y = data.data, data.target

    @Multiprocessing(n_workers=4).distribute
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
   :widths: 25 25 50

   * - Backend
     - Dependencies
     - Use case
   * - ``Multiprocessing``
     - stdlib only
     - Local parallelism on a single machine


Custom Backends
---------------

Create your own distribution backend by subclassing ``BaseDistribution``
and implementing ``_distribute``:

.. code-block:: python

    from gradient_free_optimizers.distributed import BaseDistribution

    class MyClusterBackend(BaseDistribution):
        def _distribute(self, func, params_batch):
            # Send evaluations to your cluster, collect scores
            scores = my_cluster.map(func, params_batch)
            return scores

    @MyClusterBackend(n_workers=16).distribute
    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

The ``_distribute`` method receives the original objective function and a
list of parameter dictionaries. It must return a list of scores in the
same order.


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


Limitations
-----------

The current implementation has a few constraints:

**Memory caching** is not supported with distributed evaluation. When a
distributed decorator is detected, memory is automatically disabled.

**The catch parameter** for error handling is not yet supported in
distributed mode. Worker exceptions propagate directly.

**The objective function** must be defined at module level (not a lambda
or closure) for the ``Multiprocessing`` backend on systems that do not
support the ``fork`` start method.
