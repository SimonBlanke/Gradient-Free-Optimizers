===================
Parallel Tempering
===================

Parallel Tempering (also known as Replica Exchange) runs multiple Simulated
Annealing instances at different temperatures, occasionally swapping states
between them. This allows efficient exploration at high temperatures and
exploitation at low temperatures simultaneously.


Algorithm
---------

The algorithm maintains a population of simulated annealers:

1. Each individual runs at a different temperature (cold to hot)
2. Cold individuals exploit local regions
3. Hot individuals explore broadly
4. Periodically, adjacent-temperature individuals may swap states

The swap allows good solutions found at high temperatures to be refined
at low temperatures, and stuck low-temperature individuals to escape
via high-temperature exploration.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``population``
      - int
      - 5
      - Number of temperature levels (replicas)
    * - ``n_iter_swap``
      - int
      - 5
      - Iterations between swap attempts


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ParallelTemperingOptimizer

    def multimodal(para):
        x, y = para["x"], para["y"]
        return -(np.sin(x) * np.sin(y) * np.exp(-0.01 * (x**2 + y**2)))

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = ParallelTemperingOptimizer(
        search_space,
        population=6,     # 6 temperature levels
        n_iter_swap=10,   # Swap every 10 iterations
    )

    opt.search(multimodal, n_iter=500)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Multi-modal landscapes with distinct basins
- When single Simulated Annealing gets stuck
- Problems requiring both exploration and exploitation

**Compared to other methods:**

- **vs. Single SA**: Better exploration through temperature diversity
- **vs. PSO**: Different mechanism (temperature vs. velocity)
- **vs. GA**: Continuous-focused vs. discrete-friendly


Related Algorithms
------------------

- :doc:`../local/simulated_annealing` - Single-temperature variant
- :doc:`particle_swarm` - Alternative swarm approach
- :doc:`evolution_strategy` - Population of hill climbers
