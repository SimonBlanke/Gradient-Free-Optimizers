======================
Optimization Algorithms
======================

Gradient-Free-Optimizers provides 22 optimization algorithms organized into
four categories based on their search strategy. Each algorithm has unique
characteristics that make it suitable for different types of problems.


Algorithm Overview
------------------

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Algorithm
      - Category
      - Best For
    * - :doc:`Hill Climbing <local/hill_climbing>`
      - Local
      - Smooth, unimodal functions with good starting points
    * - :doc:`Stochastic Hill Climbing <local/stochastic_hill_climbing>`
      - Local
      - Functions with minor noise or small local optima
    * - :doc:`Repulsing Hill Climbing <local/repulsing_hill_climbing>`
      - Local
      - Escaping flat regions or plateaus
    * - :doc:`Simulated Annealing <local/simulated_annealing>`
      - Local
      - Functions with multiple local optima
    * - :doc:`Downhill Simplex <local/downhill_simplex>`
      - Local
      - Low-dimensional continuous optimization
    * - :doc:`Random Search <global/random_search>`
      - Global
      - Baseline comparison, embarrassingly parallel
    * - :doc:`Grid Search <global/grid_search>`
      - Global
      - Systematic coverage of small search spaces
    * - :doc:`Random Restart Hill Climbing <global/random_restart>`
      - Global
      - Combining local refinement with global exploration
    * - :doc:`Random Annealing <global/random_annealing>`
      - Global
      - Broad initial exploration with gradual focusing
    * - :doc:`Pattern Search <global/pattern_search>`
      - Global
      - Derivative-free optimization without randomness
    * - :doc:`Powell's Method <global/powells_method>`
      - Global
      - Separable objective functions
    * - :doc:`Lipschitz Optimizer <global/lipschitz>`
      - Global
      - Functions with known smoothness bounds
    * - :doc:`DIRECT Algorithm <global/direct>`
      - Global
      - Guaranteed global convergence
    * - :doc:`Particle Swarm <population/particle_swarm>`
      - Population
      - Continuous optimization with multiple processors
    * - :doc:`Spiral Optimization <population/spiral>`
      - Population
      - Balanced exploration and exploitation
    * - :doc:`Parallel Tempering <population/parallel_tempering>`
      - Population
      - Multi-modal landscapes with distinct basins
    * - :doc:`Genetic Algorithm <population/genetic_algorithm>`
      - Population
      - Discrete/combinatorial optimization
    * - :doc:`Evolution Strategy <population/evolution_strategy>`
      - Population
      - Continuous optimization with noise
    * - :doc:`Differential Evolution <population/differential_evolution>`
      - Population
      - Continuous non-linear optimization
    * - :doc:`Bayesian Optimization <smbo/bayesian>`
      - SMBO
      - Expensive functions with continuous parameters
    * - :doc:`TPE <smbo/tpe>`
      - SMBO
      - Expensive functions, especially with conditionals
    * - :doc:`Forest Optimizer <smbo/forest>`
      - SMBO
      - Large search spaces with discrete parameters
    * - :doc:`Ensemble Optimizer <smbo/ensemble>`
      - SMBO
      - Robust surrogate modeling


Categories
----------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Local Search
        :link: local/index
        :link-type: doc
        :class-card: sd-border-start sd-border-danger

        Algorithms that explore the neighborhood of the current best solution.
        Fast and efficient for smooth functions, but may get stuck in local optima.

        **Algorithms:** Hill Climbing, Stochastic Hill Climbing, Repulsing Hill
        Climbing, Simulated Annealing, Downhill Simplex

    .. grid-item-card:: Global Search
        :link: global/index
        :link-type: doc
        :class-card: sd-border-start sd-border-success

        Algorithms designed to explore the entire search space. Better at
        avoiding local optima but may be slower to converge.

        **Algorithms:** Random Search, Grid Search, Random Restart, Random
        Annealing, Pattern Search, Powell's Method, Lipschitz, DIRECT

    .. grid-item-card:: Population-Based
        :link: population/index
        :link-type: doc
        :class-card: sd-border-start sd-border-primary

        Multiple search agents work together, sharing information about good
        regions. Natural parallelism and diverse exploration.

        **Algorithms:** Particle Swarm, Spiral, Parallel Tempering, Genetic
        Algorithm, Evolution Strategy, Differential Evolution

    .. grid-item-card:: Sequential Model-Based
        :link: smbo/index
        :link-type: doc
        :class-card: sd-border-start sd-border-warning

        Build a surrogate model of the objective function to predict promising
        regions. Ideal for expensive evaluations.

        **Algorithms:** Bayesian Optimization, TPE, Forest Optimizer, Ensemble


How to Choose
-------------

.. figure:: /_static/diagrams/algorithm_selection_flowchart.svg
    :alt: Algorithm selection flowchart
    :align: center

    Use this decision tree to narrow down the best algorithm category for your
    problem. For a more detailed guide, see :doc:`/user_guide/optimizer_selection`.

.. tip::

    **Quick Decision Guide:**

    1. **Is your objective function expensive to evaluate?**

       - Yes: Use SMBO algorithms (Bayesian, TPE, Forest)
       - No: Continue to question 2

    2. **Do you have a good starting point?**

       - Yes: Start with local search (Hill Climbing, Simulated Annealing)
       - No: Continue to question 3

    3. **Can you parallelize evaluations?**

       - Yes: Use population-based (Particle Swarm, Genetic Algorithm)
       - No: Use global search (Random Search, Pattern Search)


Computational Cost
------------------

Algorithms differ in their computational overhead (time spent in the optimizer
vs. evaluating the objective function):

.. list-table::
    :header-rows: 1

    * - Category
      - Overhead
      - Notes
    * - Local Search
      - Very Low
      - Simple neighbor generation
    * - Global Search
      - Very Low
      - Random/grid sampling
    * - Population-Based
      - Low
      - Population management overhead
    * - SMBO
      - High
      - Model training and prediction

For expensive objective functions (training ML models, running simulations),
the overhead is negligible. For cheap functions (mathematical expressions),
consider the simpler algorithms.


.. toctree::
    :maxdepth: 2
    :hidden:

    local/index
    global/index
    population/index
    smbo/index
