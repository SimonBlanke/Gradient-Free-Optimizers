======================
Differential Evolution
======================

Differential Evolution (DE) creates new candidate solutions by adding weighted
differences between population members. This self-adaptive approach automatically
scales search steps based on the current population distribution.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/differential_evolution_sphere_function_.gif
            :alt: DE on Sphere function

            **Convex function**: Population quickly converges.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/differential_evolution_ackley_function_.gif
            :alt: DE on Ackley function

            **Multi-modal function**: Difference vectors help escape
            local optima.


Algorithm
---------

For each target vector in the population:

1. **Mutation**: Create mutant vector from three random population members

   .. code-block:: text

       mutant = x_r1 + F * (x_r2 - x_r3)

   where F is the mutation factor (``mutation_rate``)

2. **Crossover**: Mix mutant with target vector

   .. code-block:: text

       trial[i] = mutant[i] if random() < CR else target[i]

   where CR is the crossover rate

3. **Selection**: Keep trial if better than target


The Key Insight
---------------

The difference vector ``(x_r2 - x_r3)`` automatically adapts:

- Early in search: population is spread out, large differences
- Late in search: population converges, small differences

This provides **self-adaptive step sizes** without explicit parameters.


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
      - 10
      - Population size
    * - ``mutation_rate``
      - float
      - 0.9
      - Mutation factor F (typically 0.5-1.0)
    * - ``crossover_rate``
      - float
      - 0.9
      - Crossover probability CR (typically 0.5-1.0)


Tuning Tips
-----------

**Mutation rate (F):**

- F = 0.5: Conservative mutations
- F = 1.0: More aggressive exploration
- F > 1.0: Very aggressive (may overshoot)

**Crossover rate (CR):**

- Low CR (0.1-0.3): More of target preserved, slower mixing
- High CR (0.7-1.0): More of mutant used, faster adaptation


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import DifferentialEvolutionOptimizer

    def rosenbrock(para):
        x, y = para["x"], para["y"]
        return -((1 - x)**2 + 100 * (y - x**2)**2)

    search_space = {
        "x": np.linspace(-5, 5, 100),
        "y": np.linspace(-5, 5, 100),
    }

    opt = DifferentialEvolutionOptimizer(
        search_space,
        population=20,
        mutation_rate=0.8,
        crossover_rate=0.9,
    )

    opt.search(rosenbrock, n_iter=300)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Continuous, non-linear optimization
- Functions with complex interactions
- When you want self-adaptive step sizes
- Real-parameter optimization

**Not ideal for:**

- Pure discrete/categorical problems
- Very small populations (needs at least 4 for basic DE)


Comparison with Other Methods
-----------------------------

.. list-table::
    :header-rows: 1

    * - Algorithm
      - Step Size
      - Mechanism
    * - DE
      - Self-adaptive (difference vectors)
      - Mutation + crossover
    * - ES
      - Mutation variance
      - Mutation + selection
    * - PSO
      - Velocity
      - Attraction to bests


Related Algorithms
------------------

- :doc:`evolution_strategy` - Alternative evolutionary approach
- :doc:`genetic_algorithm` - Discrete-focused evolution
- :doc:`particle_swarm` - Velocity-based population method
