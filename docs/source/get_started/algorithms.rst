=========================
Optimization Algorithms
=========================

Gradient-Free-Optimizers provides 22 optimization algorithms organized into four
categories. Each category represents a different strategy for exploring the search
space, and the right choice depends on your problem characteristics.

All examples below use the same objective function for comparability:

.. code-block:: python

    import numpy as np

    def objective(para):
        return -(para["x"] ** 2 + para["y"] ** 2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }


Local Search
------------

Local search algorithms start from a position and iteratively move to neighboring
positions that improve the objective. They are fast and memory-efficient, making them
a good choice when the objective landscape is smooth or when you have a reasonable
starting point.

**Algorithms:**

- ``HillClimbingOptimizer`` -- Moves to the best neighbor in each iteration
- ``StochasticHillClimbingOptimizer`` -- Accepts worse moves with a fixed probability
- ``RepulsingHillClimbingOptimizer`` -- Restarts and avoids previously visited regions
- ``SimulatedAnnealingOptimizer`` -- Accepts worse moves with decreasing probability (temperature schedule)
- ``DownhillSimplexOptimizer`` -- Nelder-Mead method using a simplex of points

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=1000)

    print(f"Best parameters: {opt.best_para}")
    print(f"Best score: {opt.best_score}")


Global Search
-------------

Global search algorithms aim to explore the entire search space rather than
focusing on a local neighborhood. They are less likely to get stuck in local optima
and provide better coverage, at the cost of slower convergence on smooth problems.

**Algorithms:**

- ``RandomSearchOptimizer`` -- Uniform random sampling across the search space
- ``GridSearchOptimizer`` -- Systematic grid-based sampling
- ``RandomRestartHillClimbingOptimizer`` -- Hill climbing with periodic random restarts
- ``RandomAnnealingOptimizer`` -- Annealing over the sampling range rather than acceptance probability
- ``PatternSearch`` -- Coordinate-based pattern moves with adaptive step sizes
- ``PowellsMethod`` -- Sequential line searches along each dimension
- ``LipschitzOptimizer`` -- Uses Lipschitz continuity assumption to guide search
- ``DirectAlgorithm`` -- Dividing Rectangles algorithm for systematic partitioning

.. code-block:: python

    from gradient_free_optimizers import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective, n_iter=1000)

    print(f"Best parameters: {opt.best_para}")
    print(f"Best score: {opt.best_score}")


Population-Based
----------------

Population-based algorithms maintain multiple candidate solutions simultaneously.
They exchange information between individuals (particles, chromosomes, agents) to
balance exploration and exploitation. These methods are well-suited for multimodal
problems with many local optima.

**Algorithms:**

- ``ParticleSwarmOptimizer`` -- Particles adjust velocity based on personal and global best
- ``EvolutionStrategyOptimizer`` -- Mutation and selection inspired by biological evolution
- ``GeneticAlgorithmOptimizer`` -- Crossover and mutation of parent solutions
- ``DifferentialEvolutionOptimizer`` -- Difference vectors between population members drive mutation
- ``SpiralOptimization`` -- Points spiral inward toward the best-known position
- ``ParallelTemperingOptimizer`` -- Multiple simulated annealing chains at different temperatures

.. code-block:: python

    from gradient_free_optimizers import ParticleSwarmOptimizer

    opt = ParticleSwarmOptimizer(search_space, population=10)
    opt.search(objective, n_iter=1000)

    print(f"Best parameters: {opt.best_para}")
    print(f"Best score: {opt.best_score}")


Sequential Model-Based
----------------------

Sequential model-based optimizers build a surrogate model of the objective function
from past evaluations and use it to decide where to evaluate next. This makes them
particularly effective when each evaluation is expensive (seconds to hours per call),
since they extract maximum information from each data point.

**Algorithms:**

- ``BayesianOptimizer`` -- Gaussian Process surrogate with acquisition function
- ``TreeStructuredParzenEstimators`` -- Models the density of good vs. bad parameters (TPE)
- ``ForestOptimizer`` -- Random forest or extra-trees surrogate model

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer

    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=50)

    print(f"Best parameters: {opt.best_para}")
    print(f"Best score: {opt.best_score}")

.. note::

    GFO ships its own Gaussian Process, Random Forest, and KDE implementations,
    so surrogate-model-based optimizers work without scikit-learn installed.
    Installing ``gradient-free-optimizers[sklearn]`` adds sklearn-based surrogates
    as an alternative backend.


Choosing an Algorithm
---------------------

As a rule of thumb:

- **Cheap evaluations, smooth landscape** -- start with ``HillClimbingOptimizer``
- **Cheap evaluations, unknown landscape** -- use ``RandomSearchOptimizer`` as a baseline, then try population-based methods
- **Expensive evaluations** -- use ``BayesianOptimizer`` or ``TreeStructuredParzenEstimators`` to minimize the number of evaluations needed
- **Need systematic coverage** -- use ``GridSearchOptimizer`` or ``DirectAlgorithm``

For detailed documentation on each algorithm including parameters and visualizations,
see the :doc:`algorithm reference </user_guide/optimizers/index>`.
