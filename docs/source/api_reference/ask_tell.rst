.. _api_reference_ask_tell:

===================
Ask/Tell Optimizers
===================

The optimizers in the ``gradient_free_optimizers.ask_tell`` subpackage
provide the same algorithms as the main package, but with a batch-capable
``ask(n=...)`` / ``tell(scores)`` interface in place of the managed
``search()`` loop. They are the right choice when you need to keep
evaluation control on your side: external worker pools, async job queues,
distributed clusters, or integration into a larger framework.

For an introduction and trade-off discussion, see
:doc:`/user_guide/ask_tell`. The constructor signatures match the
corresponding main-package optimizers, with two differences:
``initialize`` is replaced by ``initial_evaluations``, and ``nth_process``
is not exposed.


Optimizers
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    gradient_free_optimizers.ask_tell.HillClimbingOptimizer
    gradient_free_optimizers.ask_tell.StochasticHillClimbingOptimizer
    gradient_free_optimizers.ask_tell.RepulsingHillClimbingOptimizer
    gradient_free_optimizers.ask_tell.SimulatedAnnealingOptimizer
    gradient_free_optimizers.ask_tell.DownhillSimplexOptimizer
    gradient_free_optimizers.ask_tell.RandomSearchOptimizer
    gradient_free_optimizers.ask_tell.GridSearchOptimizer
    gradient_free_optimizers.ask_tell.RandomRestartHillClimbingOptimizer
    gradient_free_optimizers.ask_tell.RandomAnnealingOptimizer
    gradient_free_optimizers.ask_tell.PatternSearch
    gradient_free_optimizers.ask_tell.PowellsMethod
    gradient_free_optimizers.ask_tell.LipschitzOptimizer
    gradient_free_optimizers.ask_tell.DirectAlgorithm
    gradient_free_optimizers.ask_tell.ParticleSwarmOptimizer
    gradient_free_optimizers.ask_tell.SpiralOptimization
    gradient_free_optimizers.ask_tell.ParallelTemperingOptimizer
    gradient_free_optimizers.ask_tell.GeneticAlgorithmOptimizer
    gradient_free_optimizers.ask_tell.EvolutionStrategyOptimizer
    gradient_free_optimizers.ask_tell.DifferentialEvolutionOptimizer
    gradient_free_optimizers.ask_tell.CMAESOptimizer
    gradient_free_optimizers.ask_tell.BayesianOptimizer
    gradient_free_optimizers.ask_tell.TreeStructuredParzenEstimators
    gradient_free_optimizers.ask_tell.ForestOptimizer


Common Interface
----------------

All ask/tell optimizers expose the same methods and attributes:

.. code-block:: python

    optimizer = OptimizerClass(
        search_space,                    # dict: parameter name -> numpy array
        initial_evaluations=[            # list[tuple[dict, float]]
            (params_dict, score),
            ...,
        ],
        constraints=[],                  # optional list of constraint callables
        random_state=None,               # optional int seed
        # plus algorithm-specific parameters (epsilon, population, ...)
    )

    params_list = optimizer.ask(n=4)     # list[dict] of length n
    optimizer.tell(scores)               # list[float] of length n

    optimizer.best_score                 # float, -inf before any tell()
    optimizer.best_para                  # dict
