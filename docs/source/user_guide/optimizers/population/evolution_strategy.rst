==================
Evolution Strategy
==================

Evolution Strategy (ES) maintains a population of parent solutions and produces
offspring through mutation and optional crossover. In each generation, lambda
offspring are created by randomly perturbing the parents, all offspring are
evaluated, and the best mu individuals are selected to form the next parent
population. ES supports two selection schemes: (mu, lambda) selects parents only
from offspring, discarding all previous parents; (mu + lambda) selects from the
combined pool of parents and offspring. The mutation rate and the
offspring-to-parent ratio jointly control exploration intensity and selection
pressure.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/evolution_strategy_sphere_function_.gif
            :alt: ES on Sphere function

            **Convex function**: Population converges efficiently.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/evolution_strategy_ackley_function_.gif
            :alt: ES on Ackley function

            **Multi-modal function**: Selection pressure guides search.


ES differs from the Genetic Algorithm in this library by treating mutation as
the primary operator and crossover as secondary. This makes ES more suitable for
continuous, real-valued search spaces where small perturbations are meaningful.
The (mu, lambda) scheme is useful for noisy objective functions because it
forces re-evaluation each generation and prevents lucky outliers from persisting.
The (mu + lambda) scheme is more conservative and appropriate when evaluations
are deterministic. Compared to Differential Evolution, ES does not derive its
step sizes from population differences, so it requires more careful tuning of
mutation rate. ES is a good choice when you need explicit control over selection
pressure through the offspring-to-parent ratio.


Algorithm
---------

Each generation:

1. **Mutation**: Create offspring by mutating parents
2. **Evaluation**: Score all offspring
3. **Selection**: Select best individuals for next generation
4. **Crossover** (optional): Mix selected individuals

.. code-block:: text

    offspring = [mutate(random_parent) for _ in range(lambda)]
    if replace_parents:  # (mu, lambda)
        population = select_best(offspring, mu)
    else:                # (mu + lambda)
        population = select_best(parents + offspring, mu)

ES can use different selection schemes:

- **(mu, lambda)**: Select best mu from lambda offspring only
- **(mu + lambda)**: Select best mu from parents + offspring combined

.. note::

    The choice between (mu, lambda) and (mu + lambda) is
    significant. (mu, lambda) allows the population to "forget" bad parents,
    which is useful in noisy environments where a previously good solution
    may have been lucky. (mu + lambda) is more conservative, preserving
    proven good solutions across generations.


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
      - Population size (mu)
    * - ``offspring``
      - int
      - 20
      - Offspring per generation (lambda)
    * - ``mutation_rate``
      - float
      - 0.7
      - Probability of mutation
    * - ``crossover_rate``
      - float
      - 0.3
      - Probability of crossover
    * - ``replace_parents``
      - bool
      - False
      - If True: (mu, lambda), if False: (mu + lambda)


Selection Pressure
^^^^^^^^^^^^^^^^^^

- **High offspring/population ratio**: Strong selection, fast convergence
- **Low ratio**: Weaker selection, more diversity

.. code-block:: python

    # Strong selection (1:3 ratio)
    opt = EvolutionStrategyOptimizer(
        search_space,
        population=10,
        offspring=30,
    )

    # Weak selection (1:1.5 ratio)
    opt = EvolutionStrategyOptimizer(
        search_space,
        population=20,
        offspring=30,
    )


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import EvolutionStrategyOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = EvolutionStrategyOptimizer(
        search_space,
        population=15,
        offspring=30,
        mutation_rate=0.5,
    )

    opt.search(objective, n_iter=200)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Continuous optimization
- Noisy objective functions (robust selection)
- Problems where hill climbing variants get stuck

**Compared to GA:**

- ES: Typically continuous, mutation-focused
- GA: Often discrete, crossover-focused


3D Noisy Function Example
-------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import EvolutionStrategyOptimizer

    def noisy_sphere(para):
        x, y, z = para["x"], para["y"], para["z"]
        noise = np.random.normal(0, 0.1)
        return -(x**2 + y**2 + z**2) + noise

    search_space = {
        "x": np.linspace(-10, 10, 200),
        "y": np.linspace(-10, 10, 200),
        "z": np.linspace(-10, 10, 200),
    }

    opt = EvolutionStrategyOptimizer(
        search_space,
        population=20,
        offspring=40,
        mutation_rate=0.6,
        replace_parents=True,
    )

    opt.search(noisy_sphere, n_iter=500)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: High offspring/population ratio increases
  selection pressure (more exploitation). ``mutation_rate`` controls exploration.
- **Computational overhead**: Moderate. Producing and evaluating many offspring
  per generation adds cost.
- **Parameter sensitivity**: ``replace_parents`` is the key structural choice.
  Use ``True`` for noisy problems; ``False`` when evaluations are deterministic.


Related Algorithms
------------------

- :doc:`genetic_algorithm` - Crossover-focused evolution
- :doc:`differential_evolution` - Self-adaptive step sizes
- :doc:`particle_swarm` - Swarm-based approach
