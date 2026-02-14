===================
Parallel Tempering
===================

Parallel Tempering runs multiple Simulated Annealing instances (replicas)
simultaneously, each operating at a different temperature. High-temperature
replicas accept worse solutions frequently and explore broadly, while
low-temperature replicas accept only improvements and exploit locally. At
regular intervals, adjacent replicas attempt to swap their states based on a
Metropolis-style acceptance criterion. This swap mechanism allows promising
regions discovered by high-temperature replicas to be refined by
low-temperature replicas, and allows stuck low-temperature replicas to escape
through high-temperature exploration.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/parallel_tempering_sphere_function_.gif
            :alt: Parallel Tempering on Sphere function

            **Convex function**: Multiple replicas at different
            temperatures converge cooperatively.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/parallel_tempering_ackley_function_.gif
            :alt: Parallel Tempering on Ackley function

            **Multi-modal function**: Hot replicas explore broadly
            while cold replicas exploit locally.


Parallel Tempering directly addresses the limitation of Simulated Annealing,
which must choose a single temperature schedule that trades off exploration
against exploitation over time. By maintaining replicas across the full
temperature range simultaneously, Parallel Tempering provides both broad
exploration and local refinement at every iteration. Among the population-based
optimizers in this library, it is the natural upgrade path when Simulated
Annealing gets stuck in local optima. It works well on multi-modal landscapes
with distinct basins. The ``n_iter_swap`` parameter controls swap frequency:
too-frequent swaps disrupt local convergence, while too-infrequent swaps reduce
the benefit of the temperature ladder.


Algorithm
---------

The algorithm maintains a population of simulated annealers:

1. Each individual runs at a different temperature (cold to hot)
2. Cold individuals exploit local regions
3. Hot individuals explore broadly
4. Periodically, adjacent-temperature individuals may swap states

.. code-block:: text

    Swap criterion between replicas i and j:
    p_swap = exp((score_i - score_j) * (1/temp_i - 1/temp_j))

.. note::

    The swap mechanism creates a "temperature ladder" that
    allows good solutions to migrate from high-temperature (exploratory) replicas
    down to low-temperature (exploitative) replicas. This gives the algorithm
    both global exploration AND local precision simultaneously, unlike single
    SA which must choose between them over time.

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


3D Example with More Replicas
-----------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ParallelTemperingOptimizer

    def rastrigin_3d(para):
        A = 10
        vals = [para["x"], para["y"], para["z"]]
        return -(A * len(vals) + sum(
            v**2 - A * np.cos(2 * np.pi * v) for v in vals
        ))

    search_space = {
        "x": np.linspace(-5.12, 5.12, 200),
        "y": np.linspace(-5.12, 5.12, 200),
        "z": np.linspace(-5.12, 5.12, 200),
    }

    opt = ParallelTemperingOptimizer(
        search_space,
        population=10,
        n_iter_swap=5,
    )

    opt.search(rastrigin_3d, n_iter=1000)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: Naturally balanced through the temperature
  ladder. More replicas give finer temperature resolution.
- **Computational overhead**: Linear in the number of replicas. Each replica
  is a separate SA instance.
- **Parameter sensitivity**: ``n_iter_swap`` is critical. Too frequent swaps
  disrupt local convergence; too infrequent swaps miss the benefit of
  temperature exchange.


Related Algorithms
------------------

- :doc:`../local/simulated_annealing` - Single-temperature variant
- :doc:`particle_swarm` - Alternative swarm approach
- :doc:`evolution_strategy` - Population of hill climbers
