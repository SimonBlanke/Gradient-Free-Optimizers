=============
Random Search
=============

Random Search is the simplest optimization algorithm: it samples random positions
from the search space and tracks the best one found. Despite its simplicity, it's
often surprisingly effective and serves as an important baseline.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/random_search_sphere_function_.gif
            :alt: Random Search on Sphere function

            **Convex function**: Samples uniformly across the space.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/random_search_ackley_function_.gif
            :alt: Random Search on Ackley function

            **Multi-modal function**: No bias toward any region,
            samples the entire space.


Algorithm
---------

At each iteration:

1. Generate a random position in the search space
2. Evaluate the objective function
3. If better than current best, update best

That's it. No memory, no adaptation, no information sharing.


Why Random Search Matters
-------------------------

Random Search is more important than it might seem:

1. **Baseline comparison**: Any serious optimizer should beat Random Search.
   If it doesn't, the problem may not benefit from intelligent search.

2. **High dimensions**: In very high-dimensional spaces, random sampling
   can be surprisingly competitive with more sophisticated methods.

3. **Embarrassingly parallel**: Every evaluation is independent, making it
   trivially parallelizable.

4. **No hyperparameters**: Nothing to tune, no risk of misconfiguration.

.. note::

    Research has shown that for many hyperparameter tuning problems, Random
    Search performs nearly as well as Bayesian Optimization when given the
    same computational budget.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``search_space``
      - dict
      - required
      - Parameter space definition
    * - ``initialize``
      - dict
      - {"grid": 4, "random": 2, "vertices": 4}
      - Initialization strategy
    * - ``constraints``
      - list
      - []
      - Constraint functions
    * - ``random_state``
      - int
      - None
      - Random seed for reproducibility

Random Search has no algorithm-specific parameters.


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import RandomSearchOptimizer

    def objective(para):
        x, y = para["x"], para["y"]
        return -(x**2 + y**2)

    search_space = {
        "x": np.linspace(-10, 10, 1000),
        "y": np.linspace(-10, 10, 1000),
    }

    opt = RandomSearchOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=1000)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Establishing a baseline for comparison
- Problems with very cheap function evaluations
- Very high-dimensional spaces
- When you need trivial parallelization
- When you're not sure what optimizer to use

**Not ideal for:**

- Expensive function evaluations (wastes budget)
- When you need to converge precisely
- When you have good starting points to exploit


Comparison with Grid Search
---------------------------

.. list-table::
    :header-rows: 1

    * - Aspect
      - Random Search
      - Grid Search
    * - Coverage
      - Probabilistic
      - Deterministic
    * - High dimensions
      - Scales well
      - Curse of dimensionality
    * - Important dimensions
      - Good (projects better)
      - Poor (spreads evenly)
    * - Reproducibility
      - With random_state
      - Always identical


Related Algorithms
------------------

- :doc:`grid_search` - Systematic coverage instead of random
- :doc:`random_restart` - Random Search + local Hill Climbing
