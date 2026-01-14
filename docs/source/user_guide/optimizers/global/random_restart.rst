============================
Random Restart Hill Climbing
============================

Random Restart Hill Climbing combines local search with global exploration by
running Hill Climbing from multiple random starting points. When Hill Climbing
converges, it restarts from a new random position.


Algorithm
---------

The algorithm alternates between:

1. **Hill Climbing phase**: Local search from current position
2. **Restart phase**: Jump to a random position when stuck

This combines the efficiency of local search with the coverage of random search.


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
    )

    opt.search(rastrigin, n_iter=1000)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


Parameters
----------

This optimizer inherits all parameters from Hill Climbing:

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
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


Related Algorithms
------------------

- :doc:`../local/hill_climbing` - The local search component
- :doc:`random_search` - Pure random sampling
- :doc:`../local/simulated_annealing` - Smooth transition between explore/exploit
