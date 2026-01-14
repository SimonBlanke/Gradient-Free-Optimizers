=================
Genetic Algorithm
=================

The Genetic Algorithm (GA) is inspired by biological evolution. A population
of candidate solutions evolves through selection, crossover (recombination),
and mutation, with fitter individuals more likely to pass on their "genes."


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/genetic_algorithm_sphere_function_.gif
            :alt: GA on Sphere function

            **Convex function**: Population converges through selection.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/genetic_algorithm_ackley_function_.gif
            :alt: GA on Ackley function

            **Multi-modal function**: Diversity helps explore multiple basins.


Algorithm
---------

Each generation:

1. **Selection**: Choose parents from population (fitness-proportional)
2. **Crossover**: Combine parents to create offspring

   - Discrete recombination: randomly pick each gene from either parent
   - Blend crossover: interpolate between parent values

3. **Mutation**: Randomly perturb some genes
4. **Replacement**: New generation replaces old (or merge with elitism)


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
    * - ``offspring``
      - int
      - 10
      - Number of offspring per generation
    * - ``mutation_rate``
      - float
      - 0.5
      - Probability of mutation per gene
    * - ``crossover_rate``
      - float
      - 0.5
      - Probability of crossover vs. cloning
    * - ``crossover``
      - str
      - "discrete-recombination"
      - Crossover method
    * - ``n_parents``
      - int
      - 2
      - Number of parents per offspring


Crossover Methods
^^^^^^^^^^^^^^^^^

- **discrete-recombination**: Each gene randomly from one parent
- Can be extended to blend/arithmetic crossover for continuous parameters


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import GeneticAlgorithmOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = GeneticAlgorithmOptimizer(
        search_space,
        population=20,
        offspring=20,
        mutation_rate=0.3,
        crossover_rate=0.7,
    )

    opt.search(objective, n_iter=200)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


Tuning Tips
-----------

**Population size:**

- Larger: More diversity, slower convergence
- Smaller: Faster but may lose diversity

**Mutation rate:**

- Higher: More exploration, may disrupt good solutions
- Lower: More exploitation, may converge prematurely

**Crossover rate:**

- Higher: More recombination, explores combinations
- Lower: More cloning, preserves good individuals


When to Use
-----------

**Good for:**

- Discrete and combinatorial optimization
- Problems with complex, non-linear interactions
- When population diversity is important
- Feature selection and subset problems

**Not ideal for:**

- Simple, smooth continuous functions
- Very expensive evaluations (large population overhead)


Related Algorithms
------------------

- :doc:`evolution_strategy` - Similar but typically for continuous
- :doc:`differential_evolution` - Difference-based mutation
- :doc:`particle_swarm` - Alternative population approach
