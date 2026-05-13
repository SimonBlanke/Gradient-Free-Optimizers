===================
Boundary Strategies
===================

Many optimizers generate new candidates by perturbing an existing position.
Those raw candidates can fall outside the search-space bounds. The
``boundary`` parameter controls how GFO maps out-of-bounds candidates back
into the valid search space before the objective function sees them.

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    opt = HillClimbingOptimizer(
        search_space,
        boundary="reflect",
        random_state=1,
    )

The default is ``boundary="clip"``, which preserves the historical behavior.


Available Strategies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Strategy
     - Behavior
     - Typical use
   * - ``"clip"``
     - Clamp every out-of-bounds value to the nearest bound.
     - Conservative default when bounds are hard limits.
   * - ``"reflect"``
     - Mirror overshooting values back into the valid range.
     - Local search near edges without accumulating at the boundary.
   * - ``"periodic"``
     - Wrap values around to the other side of the range.
     - Cyclic domains such as angles, phases, or repeating schedules.
   * - ``"random"``
     - Replace only out-of-bounds coordinates with random valid values.
     - Extra exploration when large steps frequently leave the search space.
   * - ``"intermediate"``
     - Move halfway between the current position and the violated boundary.
     - Damped edge handling for local optimizers.


Dimension Handling
------------------

Boundary strategies are applied to continuous tuple dimensions and numerical
discrete dimensions:

.. code-block:: python

    import numpy as np

    search_space = {
        "learning_rate": (0.0001, 0.1),  # continuous
        "n_layers": np.arange(1, 6),     # discrete numerical
    }

Categorical dimensions are always rounded and clipped to a valid category
index because they do not have a meaningful numerical boundary geometry.

For SciPy distribution dimensions, boundary handling happens in the internal
quantile space. The objective function still receives values transformed
through the distribution ``ppf``.

Constraints are checked after boundary handling. If the repaired candidate
violates a constraint, the optimizer retries with another candidate.


Example
-------

The same optimizer can be run with different boundary strategies:

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    def objective(params):
        x = params["x"]
        y = params["y"]
        return -((x - 0.95) ** 2 + (y - 0.05) ** 2)

    search_space = {
        "x": (0.0, 1.0),
        "y": (0.0, 1.0),
    }

    for boundary in ("clip", "reflect", "periodic", "random", "intermediate"):
        opt = HillClimbingOptimizer(
            search_space,
            boundary=boundary,
            epsilon=0.6,
            n_neighbours=8,
            initialize={
                "warm_start": [{"x": 0.5, "y": 0.5}],
                "random": 2,
            },
            random_state=7,
        )
        opt.search(objective, n_iter=80, verbosity=[])

        print(boundary, opt.best_para, opt.best_score)

See ``examples/boundary_strategies.py`` for a complete runnable version.

.. note::

   ``boundary`` does not expand the search space. It only controls how
   internally generated out-of-bounds candidates are repaired before
   evaluation.
