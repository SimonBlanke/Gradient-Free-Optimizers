=========================
Stochastic Hill Climbing
=========================

Stochastic Hill Climbing extends standard Hill Climbing by adding a probability
to accept worse solutions. This helps escape shallow local optima.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/stochastic_hill_climbing_sphere_function_.gif
            :alt: Stochastic Hill Climbing on Sphere function

            **Convex function**: Still converges well, with occasional
            exploratory moves.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/stochastic_hill_climbing_ackley_function_.gif
            :alt: Stochastic Hill Climbing on Ackley function

            **Multi-modal function**: Better exploration than standard
            Hill Climbing.


Algorithm
---------

At each iteration:

1. Generate neighbors within ``epsilon`` distance
2. Evaluate neighbors
3. If a neighbor is better, move to it
4. If a neighbor is worse, move to it with probability ``p_accept``

The key insight: sometimes accepting a worse solution allows the algorithm
to escape local optima and find better regions.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``p_accept``
      - float
      - 0.5
      - Probability of accepting a worse solution
    * - ``epsilon``
      - float
      - 0.03
      - Step size as fraction of search space
    * - ``distribution``
      - str
      - "normal"
      - Step distribution
    * - ``n_neighbours``
      - int
      - 3
      - Number of neighbors per iteration


The p_accept Parameter
^^^^^^^^^^^^^^^^^^^^^^

- ``p_accept = 0.0``: Equivalent to standard Hill Climbing
- ``p_accept = 0.5``: Default, balanced exploration
- ``p_accept = 1.0``: Random walk (accepts all moves)

.. tip::

    Start with the default ``p_accept=0.5``. Decrease if the optimization
    is too erratic; increase if it gets stuck in local optima.


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import StochasticHillClimbingOptimizer

    def rastrigin(para):
        x, y = para["x"], para["y"]
        A = 10
        return -(A * 2 + (x**2 - A * np.cos(2 * np.pi * x))
                      + (y**2 - A * np.cos(2 * np.pi * y)))

    search_space = {
        "x": np.linspace(-5.12, 5.12, 100),
        "y": np.linspace(-5.12, 5.12, 100),
    }

    opt = StochasticHillClimbingOptimizer(
        search_space,
        p_accept=0.3,    # Conservative acceptance
        epsilon=0.05,
    )

    opt.search(rastrigin, n_iter=1000)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Functions with shallow local optima
- When standard Hill Climbing gets stuck
- Landscapes with noise or small perturbations

**Compared to Simulated Annealing:**

Stochastic Hill Climbing has a **constant** acceptance probability, while
Simulated Annealing **decreases** it over time. Use Simulated Annealing when
you want to explore broadly at first and exploit later.


Related Algorithms
------------------

- :doc:`hill_climbing` - No acceptance of worse solutions
- :doc:`simulated_annealing` - Decreasing acceptance probability over time
