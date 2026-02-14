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

Optimizers
----------

.. autosummary::
    :toctree: api_reference/generated/
    :template: class.rst

    gradient_free_optimizers.HillClimbingOptimizer
    gradient_free_optimizers.StochasticHillClimbingOptimizer
    gradient_free_optimizers.RepulsingHillClimbingOptimizer
    gradient_free_optimizers.SimulatedAnnealingOptimizer
    gradient_free_optimizers.DownhillSimplexOptimizer
    gradient_free_optimizers.RandomSearchOptimizer
    gradient_free_optimizers.GridSearchOptimizer
    gradient_free_optimizers.RandomRestartHillClimbingOptimizer
    gradient_free_optimizers.RandomAnnealingOptimizer
    gradient_free_optimizers.PatternSearch
    gradient_free_optimizers.PowellsMethod
    gradient_free_optimizers.LipschitzOptimizer
    gradient_free_optimizers.DirectAlgorithm
    gradient_free_optimizers.ParticleSwarmOptimizer
    gradient_free_optimizers.SpiralOptimization
    gradient_free_optimizers.ParallelTemperingOptimizer
    gradient_free_optimizers.GeneticAlgorithmOptimizer
    gradient_free_optimizers.EvolutionStrategyOptimizer
    gradient_free_optimizers.DifferentialEvolutionOptimizer
    gradient_free_optimizers.BayesianOptimizer
    gradient_free_optimizers.TreeStructuredParzenEstimators
    gradient_free_optimizers.ForestOptimizer


Common Interface
----------------

All optimizers provide these methods and attributes:


Constructor
^^^^^^^^^^^

.. code-block:: python

    optimizer = OptimizerClass(
        search_space,           # dict: parameter name -> numpy array of values
        initialize=None,        # dict: initialization strategy
        constraints=[],         # list: constraint functions
        random_state=None,      # int: random seed
        rand_rest_p=0.0,        # float: random restart probability
        nth_process=0,          # int: process ID for parallel runs
    )


search() Method
^^^^^^^^^^^^^^^

.. code-block:: python

    optimizer.search(
        objective_function,     # callable: params dict -> score
        n_iter,                 # int: number of iterations
        max_time=None,          # float: max seconds
        max_score=None,         # float: stop if score >= this
        early_stopping=None,    # dict: early stopping config
        memory=True,            # bool: cache evaluations
        memory_warm_start=None, # DataFrame: previous evaluations
        verbosity=["progress_bar", "print_results"],
        optimum="maximum",      # str: "maximum" or "minimum"
    )


Ask-Tell Interface
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Setup (required before ask/tell)
    optimizer.setup_search(objective_function, n_iter, ...)

    # Get next parameters to evaluate
    params = optimizer.ask()

    # Report evaluation result
    optimizer.tell(params, score)


Result Attributes
^^^^^^^^^^^^^^^^^

.. code-block:: python

    optimizer.best_para    # dict: best parameters found
    optimizer.best_score   # float: best score achieved
    optimizer.search_data  # DataFrame: all evaluations


Initialization Options
----------------------

The ``initialize`` parameter controls how the optimizer starts:

.. code-block:: python

    initialize = {
        "grid": 4,           # N positions on a grid
        "random": 2,         # N random positions
        "vertices": 4,       # N corner/edge positions
        "warm_start": [      # Specific starting positions
            {"x": 0.5, "y": 1.0},
        ],
    }


Stopping Conditions
-------------------

Multiple stopping conditions can be combined:

.. code-block:: python

    optimizer.search(
        objective,
        n_iter=1000,              # Max iterations
        max_time=3600,            # Max seconds
        max_score=0.99,           # Target score
        early_stopping={
            "n_iter_no_change": 50,  # Stop if no improvement
        },
    )


Verbosity Options
-----------------

Control output during search:

.. code-block:: python

    verbosity = []                    # Silent
    verbosity = ["progress_bar"]      # Progress bar only
    verbosity = ["print_results"]     # Print final results
    verbosity = ["progress_bar", "print_results"]  # Both (default)
