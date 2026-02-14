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

.. code-block:: text

    if score(neighbor) > score(current):
        accept move
    else:
        accept with probability p_accept

.. note::

    **Key Insight:** Unlike Simulated Annealing where acceptance probability
    **decreases** over time via a temperature schedule, Stochastic Hill Climbing
    uses a **constant** acceptance probability throughout the entire search. This
    means it never fully transitions to pure exploitation, providing continuous
    escape capability at the cost of convergence precision.


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


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import StochasticHillClimbingOptimizer

    def ackley_3d(para):
        import math
        vals = [para["x"], para["y"], para["z"]]
        n = len(vals)
        sum_sq = sum(v**2 for v in vals) / n
        sum_cos = sum(np.cos(2 * math.pi * v) for v in vals) / n
        return -(- 20 * np.exp(-0.2 * np.sqrt(sum_sq))
                 - np.exp(sum_cos) + 20 + math.e)

    search_space = {
        "x": np.linspace(-5, 5, 200),
        "y": np.linspace(-5, 5, 200),
        "z": np.linspace(-5, 5, 200),
    }

    opt = StochasticHillClimbingOptimizer(
        search_space,
        p_accept=0.2,
        epsilon=0.05,
        n_neighbours=5,
    )

    opt.search(ackley_3d, n_iter=2000)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: ``p_accept`` directly controls this balance.
  Higher values give more exploration but slower convergence.
- **Computational overhead**: Same as Hill Climbing (minimal).
- **Parameter sensitivity**: The ``p_accept`` parameter is critical. Values near 1.0
  degrade to a random walk; values near 0.0 reduce to standard Hill Climbing.


Related Algorithms
------------------

- :doc:`hill_climbing` - No acceptance of worse solutions
- :doc:`simulated_annealing` - Decreasing acceptance probability over time
