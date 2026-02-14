=================
Genetic Algorithm
=================

The Genetic Algorithm (GA) evolves a population of candidate solutions through
repeated application of selection, crossover (recombination), and mutation.
In each generation, parents are selected with probability proportional to their
fitness. Crossover combines parameters from two parents to produce offspring,
while mutation randomly perturbs individual parameters. The offspring replace
some or all of the current population, and the cycle repeats. The crossover
operator is the central mechanism: it assembles new candidates from partial
solutions distributed across different individuals in the population.


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


Among the evolutionary algorithms in this library, GA is distinguished by its
reliance on crossover as the primary search operator. This makes it well suited
for discrete and categorical search spaces where the optimal solution is a
combination of favorable values spread across multiple candidates. Evolution
Strategy and Differential Evolution, by contrast, are mutation-driven and
designed for continuous parameters. GA is the preferred choice for feature
selection, combinatorial problems, and mixed-type search spaces. On purely
continuous landscapes, Differential Evolution or PSO will typically converge
faster because their operators are tailored to real-valued search.


Algorithm
---------

Each generation:

1. **Selection**: Choose parents from population (fitness-proportional)
2. **Crossover**: Combine parents to create offspring

   - Discrete recombination: randomly pick each gene from either parent
   - Blend crossover: interpolate between parent values

3. **Mutation**: Randomly perturb some genes
4. **Replacement**: New generation replaces old (or merge with elitism)

.. code-block:: text

    parents = select(population, fitness_proportional)
    child = crossover(parent1, parent2, rate=crossover_rate)
    child = mutate(child, rate=mutation_rate)
    population = replace(population, child)

.. note::

    GA's crossover operation is what distinguishes it from
    other evolutionary methods. By combining "genes" from two good parents,
    crossover can discover solutions that neither parent contained. This is
    particularly powerful for discrete/combinatorial problems where the optimal
    solution is a specific combination of features from different regions of
    the search space.


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


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import GeneticAlgorithmOptimizer

    def feature_selection_proxy(para):
        """Simulates a feature selection objective."""
        features = [para[f"f{i}"] for i in range(6)]
        # Prefer solutions with fewer active features but good coverage
        active = sum(1 for f in features if f > 0.5)
        quality = sum(np.sin(f * 3) for f in features)
        return quality - 0.5 * active

    search_space = {
        f"f{i}": np.array([0.0, 1.0])
        for i in range(6)
    }

    opt = GeneticAlgorithmOptimizer(
        search_space,
        population=30,
        offspring=30,
        mutation_rate=0.2,
        crossover_rate=0.8,
    )

    opt.search(feature_selection_proxy, n_iter=500)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: ``mutation_rate`` drives exploration;
  ``crossover_rate`` combines existing solutions (exploitation of known
  good regions). Population size provides baseline diversity.
- **Computational overhead**: Moderate. Selection, crossover, and mutation
  all add overhead per generation.
- **Parameter sensitivity**: The balance between mutation and crossover
  rates is critical. Too much mutation destroys good solutions; too little
  leads to premature convergence.


Related Algorithms
------------------

- :doc:`evolution_strategy` - Similar but typically for continuous
- :doc:`differential_evolution` - Difference-based mutation
- :doc:`particle_swarm` - Alternative population approach
