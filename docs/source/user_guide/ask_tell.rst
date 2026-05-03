==================
Ask/Tell Interface
==================

The ask/tell interface inverts control of the optimization loop. Instead of
handing your objective function to ``search()`` and letting GFO drive
evaluation, you call ``ask()`` to get parameter sets, evaluate them yourself
in whatever way suits your environment, and call ``tell()`` to feed the
scores back. The optimizer's algorithmic logic is identical to the
managed-loop interface; only the orchestration changes.

This is the right interface when you need to keep evaluation under your own
control: parallel pools you already manage, asynchronous job queues, distributed
clusters with their own scheduling, or integration into a larger optimization
framework that wants to combine GFO algorithms with its own logic.


.. grid:: 1

   .. grid-item-card::
      :class-card: sd-border-primary gfo-compact

      .. code-block:: python

          opt = Optimizer(search_space, initial_evaluations=[...])
          params = opt.ask(n=4)             # 1. Propose
          scores = [evaluate(p) for p in params]   # 2. Evaluate (your code)
          opt.tell(scores)                  # 3. Report back


When to Use Which Interface
---------------------------

Both interfaces share the same 23 single-objective algorithms. The choice is
about who runs the loop, not which optimizer you get.

.. list-table::
   :header-rows: 1
   :widths: 35 32 33

   * - Aspect
     - ``search()``
     - ask/tell
   * - Loop control
     - GFO drives evaluation
     - You drive evaluation
   * - Initialization
     - Built-in strategies (grid, random, vertices, warm_start)
     - You supply ``initial_evaluations``
   * - Evaluation caching
     - ``memory`` parameter, in-process or persistent
     - You handle caching yourself
   * - Progress and summary output
     - ``verbosity`` flags, progress bar, summary box
     - None
   * - Stopping conditions
     - ``n_iter``, ``max_time``, ``max_score``, ``early_stopping``
     - You decide when to stop the loop
   * - Best result access
     - ``opt.best_score``, ``opt.best_para``, ``opt.search_data``
     - ``opt.best_score``, ``opt.best_para``
   * - Per-iteration results DataFrame
     - ``opt.search_data``
     - Not available
   * - Constraints
     - Supported
     - Supported
   * - Random state
     - Supported
     - Supported

Use ``search()`` when you have a self-contained Python function and want GFO
to handle everything around evaluation. Use ask/tell when evaluation lives
outside Python's normal call flow, when you already have an evaluation
backend you want to keep, or when you are embedding GFO algorithms in a
larger system.


Basic Usage
-----------

The constructor requires ``initial_evaluations``: a list of
``(parameter_dict, score)`` pairs that seed the optimizer. After
construction the optimizer is in iteration state and ``ask`` / ``tell``
can be called.

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers.ask_tell import HillClimbingOptimizer

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    def objective(params):
        return -(params["x"] ** 2 + params["y"] ** 2)

    initial_evaluations = [
        ({"x": 0.5, "y": 1.0}, objective({"x": 0.5, "y": 1.0})),
        ({"x": -3.0, "y": 0.0}, objective({"x": -3.0, "y": 0.0})),
    ]

    opt = HillClimbingOptimizer(
        search_space,
        initial_evaluations=initial_evaluations,
    )

    for _ in range(50):
        params = opt.ask(n=1)[0]
        score = objective(params)
        opt.tell([score])

    print(opt.best_para)
    print(opt.best_score)


Initial Evaluations
-------------------

The ``initial_evaluations`` argument replaces the ``initialize`` strategies
(``grid``, ``random``, ``vertices``, ``warm_start``) used by ``search()``. You
provide the seed evaluations directly because the ask/tell interface has no
way to call your objective function on its own.

For local-search and SMBO algorithms, a single evaluation is enough to start.
Population-based optimizers (Particle Swarm, Genetic Algorithm, Evolution
Strategy, Differential Evolution, Parallel Tempering, Spiral Optimization,
CMA-ES) need at least one evaluation per population member, since each
sub-optimizer requires a starting point. The constructor enforces this and
raises ``ValueError`` if too few evaluations are supplied.

.. code-block:: python

    from gradient_free_optimizers.ask_tell import ParticleSwarmOptimizer

    population = 10

    initial_evaluations = [
        (params, objective(params))
        for params in random_starting_points(n=population)
    ]

    opt = ParticleSwarmOptimizer(
        search_space,
        initial_evaluations=initial_evaluations,
        population=population,
    )


Batch ask/tell
--------------

``ask(n=k)`` returns a list of ``k`` parameter dictionaries. ``tell`` expects
a list of ``k`` scores in the same order. The batch size you pass to ``ask``
is independent of any internal algorithm parameter (population size,
neighbour count). For SMBO optimizers (Bayesian, TPE, Forest), batch
proposals are diversified internally so a single ``ask(n=8)`` does not return
eight near-identical points clustered around the acquisition peak.

Each ``ask`` must be followed by exactly one ``tell`` before the next
``ask``. Calling ``ask`` twice in a row raises ``RuntimeError``; calling
``tell`` with the wrong number of scores raises ``ValueError``.

.. code-block:: python

    from gradient_free_optimizers.ask_tell import BayesianOptimizer
    from concurrent.futures import ProcessPoolExecutor

    opt = BayesianOptimizer(
        search_space,
        initial_evaluations=initial_evaluations,
    )

    with ProcessPoolExecutor(max_workers=4) as pool:
        for _ in range(25):
            params_batch = opt.ask(n=4)
            scores = list(pool.map(objective, params_batch))
            opt.tell(scores)


Constraints
-----------

Constraint functions work the same way as in ``search()``. ``ask`` only
returns parameter sets that satisfy all constraints, so your evaluation
code does not need its own constraint check.

.. code-block:: python

    def inside_circle(params):
        return params["x"] ** 2 + params["y"] ** 2 <= 25

    opt = HillClimbingOptimizer(
        search_space,
        initial_evaluations=initial_evaluations,
        constraints=[inside_circle],
    )


Limitations
-----------

The ask/tell interface trades convenience for control. Several features that
``search()`` provides have no counterpart here, by design:

The optimizer keeps no per-iteration history beyond what the algorithm needs
internally. There is no ``opt.search_data`` DataFrame, no convergence list,
no timing breakdown. If you want any of that, record it yourself in the loop.

There is no evaluation caching. If your objective is deterministic and you
expect duplicate proposals, deduplicate them before evaluation or wrap your
objective with ``functools.lru_cache``.

There is no progress bar, no print_summary, no verbosity option. The
optimizer does its work silently.

There are no built-in stopping conditions beyond your own loop. ``max_time``,
``max_score``, ``early_stopping`` from ``search()`` are not exposed; you
implement them in your loop.

There are no callbacks and no ``catch`` parameter. Exception handling is your
responsibility.

Multi-objective optimizers (NSGA-II, MOEA/D, SMS-EMOA) are not exposed
through ask/tell. Their score semantics differ from the single-objective
case, and a clean ask/tell shape for them needs separate design.


Available Optimizers
--------------------

All 23 single-objective algorithms from the main package are available with
identical algorithmic behaviour:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Category
     - Optimizers
   * - Local
     - ``HillClimbingOptimizer``, ``StochasticHillClimbingOptimizer``,
       ``RepulsingHillClimbingOptimizer``, ``SimulatedAnnealingOptimizer``,
       ``DownhillSimplexOptimizer``
   * - Global
     - ``RandomSearchOptimizer``, ``GridSearchOptimizer``,
       ``RandomRestartHillClimbingOptimizer``, ``RandomAnnealingOptimizer``,
       ``PatternSearch``, ``PowellsMethod``, ``LipschitzOptimizer``,
       ``DirectAlgorithm``
   * - Population
     - ``ParticleSwarmOptimizer``, ``SpiralOptimization``,
       ``ParallelTemperingOptimizer``, ``GeneticAlgorithmOptimizer``,
       ``EvolutionStrategyOptimizer``, ``DifferentialEvolutionOptimizer``,
       ``CMAESOptimizer``
   * - SMBO
     - ``BayesianOptimizer``, ``TreeStructuredParzenEstimators``,
       ``ForestOptimizer``

All ask/tell optimizers live in the ``gradient_free_optimizers.ask_tell``
subpackage. The constructor signatures match the corresponding main-package
optimizers, with two differences: ``initialize`` is replaced by
``initial_evaluations``, and ``nth_process`` is not exposed.
