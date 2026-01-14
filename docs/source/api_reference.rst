.. _api_reference:

=============
API Reference
=============

Complete API documentation for all optimizers and their methods.

----

Overview
========

All 22 optimizers in Gradient-Free-Optimizers share a common interface,
making it easy to switch between algorithms without changing your code.

.. code-block:: python

    # All optimizers follow this pattern
    from gradient_free_optimizers import SomeOptimizer

    opt = SomeOptimizer(
        search_space,          # Required: dict of parameter arrays
        initialize=None,       # Optional: initialization strategy
        constraints=[],        # Optional: constraint functions
        random_state=None,     # Optional: random seed for reproducibility
        rand_rest_p=0.0,       # Optional: random restart probability
    )

    opt.search(
        objective_function,    # Your function to maximize
        n_iter=100,           # Number of iterations
        max_time=None,        # Optional: max seconds
        max_score=None,       # Optional: stop when score >= target
        early_stopping=None,  # Optional: stop when progress stalls
        memory=True,          # Cache evaluations
        memory_warm_start=None,  # Continue from previous run
        verbosity=["progress_bar", "print_results"],
        optimum="maximum",    # "maximum" or "minimum"
    )

    # Results available after search
    best_params = opt.best_para
    best_score = opt.best_score
    all_evaluations = opt.search_data

----

Optimizer Categories
====================

.. grid:: 2 2 2 2
   :gutter: 3

   .. grid-item-card:: Local Search
      :class-card: sd-border-start sd-border-danger

      - :class:`~gradient_free_optimizers.HillClimbingOptimizer`
      - :class:`~gradient_free_optimizers.StochasticHillClimbingOptimizer`
      - :class:`~gradient_free_optimizers.RepulsingHillClimbingOptimizer`
      - :class:`~gradient_free_optimizers.SimulatedAnnealingOptimizer`
      - :class:`~gradient_free_optimizers.DownhillSimplexOptimizer`

   .. grid-item-card:: Global Search
      :class-card: sd-border-start sd-border-success

      - :class:`~gradient_free_optimizers.RandomSearchOptimizer`
      - :class:`~gradient_free_optimizers.GridSearchOptimizer`
      - :class:`~gradient_free_optimizers.RandomRestartHillClimbingOptimizer`
      - :class:`~gradient_free_optimizers.RandomAnnealingOptimizer`
      - :class:`~gradient_free_optimizers.PatternSearch`
      - :class:`~gradient_free_optimizers.PowellsMethod`
      - :class:`~gradient_free_optimizers.LipschitzOptimizer`
      - :class:`~gradient_free_optimizers.DirectAlgorithm`

   .. grid-item-card:: Population-Based
      :class-card: sd-border-start sd-border-primary

      - :class:`~gradient_free_optimizers.ParticleSwarmOptimizer`
      - :class:`~gradient_free_optimizers.SpiralOptimization`
      - :class:`~gradient_free_optimizers.ParallelTemperingOptimizer`
      - :class:`~gradient_free_optimizers.GeneticAlgorithmOptimizer`
      - :class:`~gradient_free_optimizers.EvolutionStrategyOptimizer`
      - :class:`~gradient_free_optimizers.DifferentialEvolutionOptimizer`

   .. grid-item-card:: Sequential Model-Based
      :class-card: sd-border-start sd-border-warning

      - :class:`~gradient_free_optimizers.BayesianOptimizer`
      - :class:`~gradient_free_optimizers.TreeStructuredParzenEstimators`
      - :class:`~gradient_free_optimizers.ForestOptimizer`

----

.. toctree::
   :maxdepth: 2
   :hidden:

   api_reference/index
