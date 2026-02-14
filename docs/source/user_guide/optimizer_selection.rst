===================
Optimizer Selection
===================

Choosing the right optimizer depends on your problem characteristics. This
guide helps you select the best algorithm for your use case.


.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Local Search
      :class-card: gfo-compact gfo-local

      Hill Climbing, Simulated Annealing,
      Downhill Simplex. Best for exploiting
      promising regions.

   .. grid-item-card:: Global Search
      :class-card: gfo-compact gfo-global

      Random Search, Grid Search, Pattern
      Search, DIRECT. Best for broad
      exploration.

   .. grid-item-card:: Population-Based
      :class-card: gfo-compact gfo-population

      Particle Swarm, Genetic Algorithm,
      Evolution Strategy. Best for
      parallel evaluation.

   .. grid-item-card:: Sequential Model-Based
      :class-card: gfo-compact gfo-smbo

      Bayesian, TPE, Forest, Ensemble.
      Best for expensive objective functions.


Decision Tree
-------------

.. code-block:: text

    Is your objective function expensive (> 1 second)?
    ├── Yes → Use SMBO (Bayesian, TPE, Forest)
    │         └── Continuous params? → BayesianOptimizer
    │         └── Many categoricals? → TreeStructuredParzenEstimators
    │         └── Many iterations? → ForestOptimizer
    │
    └── No → Continue...

    Do you have a good starting point?
    ├── Yes → Use Local Search
    │         └── Smooth function? → HillClimbingOptimizer
    │         └── Multiple local optima? → SimulatedAnnealingOptimizer
    │
    └── No → Continue...

    Can you evaluate in parallel?
    ├── Yes → Use Population-Based
    │         └── Continuous? → ParticleSwarmOptimizer
    │         └── Discrete? → GeneticAlgorithmOptimizer
    │
    └── No → Use Global Search
              └── Need baseline? → RandomSearchOptimizer
              └── Small space? → GridSearchOptimizer
              └── Unknown landscape? → DirectAlgorithm


By Problem Type
---------------

**Hyperparameter Tuning (ML)**

- First choice: ``BayesianOptimizer`` (efficient with expensive training)
- With many categoricals: ``TreeStructuredParzenEstimators``
- Large search spaces: ``ForestOptimizer``
- Quick baseline: ``RandomSearchOptimizer``


**Continuous Optimization**

- Smooth, unimodal: ``HillClimbingOptimizer``
- Multi-modal: ``SimulatedAnnealingOptimizer`` or ``ParticleSwarmOptimizer``
- Need global guarantee: ``DirectAlgorithm``


**Discrete/Combinatorial**

- Feature selection: ``GeneticAlgorithmOptimizer``
- Mixed spaces: ``TreeStructuredParzenEstimators``
- Small discrete space: ``GridSearchOptimizer``


**Unknown Landscape**

- Start with ``RandomSearchOptimizer`` to explore
- Then refine with ``BayesianOptimizer`` or ``SimulatedAnnealingOptimizer``


By Evaluation Budget
--------------------

.. list-table::
    :header-rows: 1

    * - Budget
      - Recommended Optimizers
    * - < 50 iterations
      - Bayesian, TPE (learn quickly)
    * - 50-200 iterations
      - Bayesian, Forest, SimulatedAnnealing
    * - 200-1000 iterations
      - Forest, ParticleSwarm, EvolutionStrategy
    * - > 1000 iterations
      - Any (cheaper algorithms work well)


By Function Cost
----------------

**Very cheap (< 0.01s per evaluation)**

- Use simple algorithms: Hill Climbing, Random Search
- SMBO overhead may dominate

**Cheap (0.01s - 1s)**

- Most algorithms work well
- Population-based gives good coverage

**Expensive (1s - 60s)**

- SMBO algorithms (Bayesian, TPE, Forest)
- Enable memory caching

**Very expensive (> 60s)**

- Bayesian Optimization is most sample-efficient
- Consider parallelization with Hyperactive


Algorithm Characteristics
-------------------------

.. list-table::
    :header-rows: 1
    :widths: 20 20 20 40

    * - Algorithm
      - Exploration
      - Exploitation
      - Notes
    * - Hill Climbing
      - Low
      - High
      - Fast, may get stuck
    * - Simulated Annealing
      - Medium
      - High
      - Good general-purpose
    * - Random Search
      - High
      - Low
      - Great baseline
    * - Particle Swarm
      - Medium
      - Medium
      - Good for parallelism
    * - Genetic Algorithm
      - High
      - Medium
      - Good for discrete
    * - Bayesian
      - Adaptive
      - Adaptive
      - Best for expensive
    * - TPE
      - Adaptive
      - Adaptive
      - Handles categoricals well


Starting Recommendation
-----------------------

If unsure, this progression often works well:

1. **RandomSearchOptimizer** (10% of budget) - establish baseline
2. **BayesianOptimizer** or **SimulatedAnnealingOptimizer** (90% of budget) - optimize

Or for very limited budgets:

1. **BayesianOptimizer** with ``xi=0.1`` (higher exploration) for first half
2. **BayesianOptimizer** with ``xi=0.01`` (higher exploitation) for second half
