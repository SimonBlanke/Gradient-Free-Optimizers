=============================
Population-Based Algorithms
=============================

Population-based algorithms maintain a collection of candidate solutions (individuals)
that evolve together. They share information about promising regions and provide
natural parallelism for multi-core systems.


Overview
--------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Algorithm
      - Description
    * - :doc:`particle_swarm`
      - Particles move toward best positions with velocity dynamics.
    * - :doc:`spiral`
      - Particles spiral inward toward the global best.
    * - :doc:`parallel_tempering`
      - Multiple simulated annealers at different temperatures.
    * - :doc:`genetic_algorithm`
      - Selection, crossover, and mutation inspired by evolution.
    * - :doc:`evolution_strategy`
      - Population of hill climbers with occasional mixing.
    * - :doc:`differential_evolution`
      - Creates new solutions from weighted differences.


When to Use Population-Based
----------------------------

**Good for:**

- Multi-modal landscapes with distinct basins
- Problems that benefit from parallelization
- Discrete and combinatorial optimization (GA, DE)
- Robust optimization in noisy environments

**Not ideal for:**

- Very expensive objective functions (each iteration evaluates N individuals)
- Simple unimodal functions (overkill)
- Very tight computational budgets


Common Parameters
-----------------

All population-based algorithms share:

.. list-table::
    :header-rows: 1
    :widths: 20 15 65

    * - Parameter
      - Default
      - Description
    * - ``population``
      - 10
      - Number of individuals in the population


Population Size
^^^^^^^^^^^^^^^

Larger populations provide better coverage but require more evaluations per iteration:

.. code-block:: python

    from gradient_free_optimizers import ParticleSwarmOptimizer

    # Small population - faster iterations, may miss optima
    opt = ParticleSwarmOptimizer(search_space, population=5)

    # Large population - thorough coverage, slower iterations
    opt = ParticleSwarmOptimizer(search_space, population=50)

.. tip::

    **Rule of thumb:** Use ``population = 10 * n_dimensions`` as a starting point,
    then adjust based on convergence behavior.


Algorithm Comparison
--------------------

.. list-table::
    :header-rows: 1
    :widths: 20 20 20 40

    * - Algorithm
      - Information Sharing
      - Best For
      - Key Feature
    * - Particle Swarm
      - Global + Personal best
      - Continuous optimization
      - Velocity-based movement
    * - Spiral
      - Global best
      - Balanced exploration
      - Spiral trajectories
    * - Parallel Tempering
      - Temperature swaps
      - Multi-modal
      - Temperature diversity
    * - Genetic Algorithm
      - Crossover
      - Discrete/combinatorial
      - Evolutionary operators
    * - Evolution Strategy
      - Selection
      - Noisy optimization
      - Robust selection
    * - Differential Evolution
      - Difference vectors
      - Non-linear continuous
      - Self-adaptive steps


Visualization
-------------

.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/particle_swarm_optimization_sphere_function_.gif
            :alt: Particle Swarm on Sphere function

            Particle Swarm: particles converge toward the best known position.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/genetic_algorithm_sphere_function_.gif
            :alt: Genetic Algorithm on Sphere function

            Genetic Algorithm: population evolves through selection and crossover.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/evolution_strategy_sphere_function_.gif
            :alt: Evolution Strategy on Sphere function

            Evolution Strategy: population of hill climbers with selection.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/differential_evolution_sphere_function_.gif
            :alt: Differential Evolution on Sphere function

            Differential Evolution: creates trials from weighted differences.


Example: Particle Swarm Optimization
------------------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ParticleSwarmOptimizer

    def rastrigin(para):
        x = para["x"]
        y = para["y"]
        A = 10
        return -(A * 2 + (x**2 - A * np.cos(2 * np.pi * x))
                      + (y**2 - A * np.cos(2 * np.pi * y)))

    search_space = {
        "x": np.linspace(-5.12, 5.12, 100),
        "y": np.linspace(-5.12, 5.12, 100),
    }

    opt = ParticleSwarmOptimizer(
        search_space,
        population=20,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
    )

    opt.search(rastrigin, n_iter=200)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


Algorithms
----------

.. toctree::
    :maxdepth: 1

    particle_swarm
    spiral
    parallel_tempering
    genetic_algorithm
    evolution_strategy
    differential_evolution
