============================
Random Restart Hill Climbing
============================

Random Restart Hill Climbing combines local search with global exploration by
running Hill Climbing from multiple random starting points. When Hill Climbing
converges, it restarts from a new random position.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/random_restart_hill_climbing_sphere_function_.gif
            :alt: Random Restart Hill Climbing on Sphere function

            **Convex function**: Hill Climbing phases converge quickly,
            restarts explore different regions.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/random_restart_hill_climbing_ackley_function_.gif
            :alt: Random Restart Hill Climbing on Ackley function

            **Multi-modal function**: Multiple restarts explore
            different basins of attraction.


Algorithm
---------

The algorithm alternates between:

1. **Hill Climbing phase**: Local search from current position for ``n_iter_restart`` iterations
2. **Restart phase**: Jump to a random position

.. code-block:: text

    for each iteration:
        if iter % n_iter_restart == 0:
            pos = random_position(search_space)  # Restart
        else:
            pos = hill_climbing_step(pos)         # Local search

.. note::

    **Key Insight:** Random Restart HC provides a hard boundary between
    exploration and exploitation. Each restart is a completely independent
    local search, which means the algorithm naturally samples from multiple
    basins. The ``n_iter_restart`` parameter controls the balance: shorter
    restarts give more global coverage, longer restarts give better local
    convergence within each basin.


When to Use
-----------

**Good for:**

- Multi-modal functions with distinct basins
- When Hill Climbing gets stuck in local optima
- Problems where local optima are well-separated

**Compared to other approaches:**

- **vs. Pure Random Search**: More efficient within each basin
- **vs. Simulated Annealing**: Sharper transitions between explore/exploit
- **vs. Repulsing HC**: Completely new starts instead of larger steps


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import RandomRestartHillClimbingOptimizer

    def rastrigin(para):
        x, y = para["x"], para["y"]
        A = 10
        return -(A * 2 + (x**2 - A * np.cos(2 * np.pi * x))
                      + (y**2 - A * np.cos(2 * np.pi * y)))

    search_space = {
        "x": np.linspace(-5.12, 5.12, 100),
        "y": np.linspace(-5.12, 5.12, 100),
    }

    opt = RandomRestartHillClimbingOptimizer(
        search_space,
        epsilon=0.03,
        n_neighbours=3,
        n_iter_restart=10,
    )

    opt.search(rastrigin, n_iter=1000)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``n_iter_restart``
      - int
      - 10
      - Number of iterations between random restarts
    * - ``epsilon``
      - float
      - 0.03
      - Step size for Hill Climbing phase
    * - ``distribution``
      - str
      - "normal"
      - Step distribution
    * - ``n_neighbours``
      - int
      - 3
      - Neighbors per Hill Climbing iteration


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import RandomRestartHillClimbingOptimizer

    def schwefel_3d(para):
        vals = [para["x"], para["y"], para["z"]]
        return sum(
            v * np.sin(np.sqrt(abs(v))) for v in vals
        )

    search_space = {
        "x": np.linspace(-500, 500, 500),
        "y": np.linspace(-500, 500, 500),
        "z": np.linspace(-500, 500, 500),
    }

    opt = RandomRestartHillClimbingOptimizer(
        search_space,
        epsilon=0.05,
        n_neighbours=5,
        n_iter_restart=20,
    )

    opt.search(schwefel_3d, n_iter=3000)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: Controlled by ``n_iter_restart``. Short
  restart intervals give more exploration (more basins sampled); long intervals
  give deeper exploitation per basin.
- **Computational overhead**: Same as Hill Climbing (minimal).
- **Parameter sensitivity**: ``n_iter_restart`` is the critical parameter. Too
  short means Hill Climbing never converges locally; too long means fewer
  basins are explored.


Related Algorithms
------------------

- :doc:`../local/hill_climbing` - The local search component
- :doc:`random_search` - Pure random sampling
- :doc:`../local/simulated_annealing` - Smooth transition between explore/exploit
