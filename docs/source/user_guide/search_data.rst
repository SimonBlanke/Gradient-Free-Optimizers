.. _search_data:

=====================
Search Data & Summary
=====================

After running ``search()``, GFO provides two ways to understand what happened:
a **CLI summary** printed directly to the terminal, and a programmatic
**data accessor** (``opt.data``) for analysis in code.


Quick Start
-----------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def objective(params):
        return -(params["x"]**2 + params["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = HillClimbingOptimizer(search_space)

    # Default verbosity includes print_results and print_times
    opt.search(objective, n_iter=50)

    # Access data programmatically
    print(opt.data.best_score)
    print(opt.data.convergence_data)

This prints a formatted summary box to the terminal:

.. code-block:: text

    ┌─ Search Summary ────────────────────────────┐
    │                                             │
    │  Objective:          objective              │
    │  Optimizer:          HillClimbingOptimizer  │
    │  Random state:       206805842              │
    │                                             │
    │  ── Results ──────────────────────────────  │
    │  Best score:         -0.0308                │
    │  Best iter:          2                      │
    │  Best parameters:                           │
    │    x:                -0.101                 │
    │    y:                -0.101                 │
    │                                             │
    │  ── Search ───────────────────────────────  │
    │  Iterations:         50                     │
    │    Initialization:   10 (20.0%)             │
    │    Optimization:     40 (80.0%)             │
    │  Improvements:       2                      │
    │  Accepted:           14/50 (28.0%)          │
    │  Last improvement:   iter 2                 │
    │  Longest plateau:    47 iterations          │
    │                                             │
    │  ── Score Statistics ─────────────────────  │
    │  Min:                -225                   │
    │  Max:                -0.03082               │
    │  Mean:               -24.83                 │
    │  Std:                62.24                  │
    │                                             │
    │  ── Timing ───────────────────────────────  │
    │  Evaluation time:    0.000s (0.0%)          │
    │  Optimization time:  0.002s (100.0%)        │
    │  Iteration time:     0.002s                 │
    │  Throughput:         26810.94 iter/sec      │
    │                                             │
    └─────────────────────────────────────────────┘

----


Search Summary (CLI)
--------------------

The summary box is controlled by ``verbosity`` flags. The default
``["progress_bar", "print_results", "print_times"]`` shows the Results and
Timing sections. Add more flags to show additional sections:

.. code-block:: python

    # Default: Results + Timing
    opt.search(objective, n_iter=100)

    # All four sections
    opt.search(objective, n_iter=100, verbosity=[
        "progress_bar", "print_results", "print_times",
        "print_search_stats", "print_statistics",
    ])

    # Summary only, no progress bar
    opt.search(objective, n_iter=100, verbosity=["print_results", "print_times"])

    # Completely silent
    opt.search(objective, n_iter=100, verbosity=False)


Summary Sections
^^^^^^^^^^^^^^^^

**General** -- Identifies the run for reproducibility.

- **Objective**: Name of the objective function
- **Optimizer**: Optimizer class used
- **Random state**: Random seed (pass ``random_state`` to the constructor to fix it)

**Results** -- What the optimizer found.

- **Best score**: Highest score achieved
- **Best iter**: Iteration where the best score was found
- **Best parameters**: Parameter values at the best score

**Search** -- How the optimizer explored.

- **Iterations**: Total iterations, split into initialization and optimization phases
- **Improvements**: How many times the best score improved
- **Accepted**: Positions accepted out of total proposed. A low rate means the
  optimizer is selective (e.g. hill climbing rejects worse positions)
- **Last improvement**: Iteration of the last score improvement. If this is much
  lower than the total iterations, you may be running too many iterations.
- **Longest plateau**: Longest stretch without any improvement

**Score Statistics** -- Distribution of all scores evaluated.

- **Min / Max**: Score range (excludes ``inf`` and ``nan``)
- **Mean / Std**: Mean and standard deviation of scores

**Timing** -- Where time was spent.

- **Evaluation time**: Time inside the objective function (with percentage)
- **Optimization time**: Time in optimizer logic (total minus evaluation)
- **Iteration time**: Total wall time
- **Throughput**: Iterations per second (or seconds per iteration if slow)


CLI Help Tool
^^^^^^^^^^^^^

For a quick reference in the terminal, run:

.. code-block:: bash

    gfo-help

This prints a compact reference of all summary fields with their descriptions
and corresponding ``opt.data`` property names.

----


``opt.data`` Properties
-----------------------

After calling ``search()``, the ``opt.data`` property returns a
:class:`~gradient_free_optimizers._data.data_accessor.DataAccessor` object with
computed metrics. All properties are read-only.


Results
^^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``best_score``
     - ``float``
     - Highest score found during the search.
   * - ``best_para``
     - ``dict``
     - Parameter values corresponding to the best score.
   * - ``best_iteration``
     - ``int``
     - Iteration index where the best score was found.


Iteration Counts
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``n_iter``
     - ``int``
     - Total iterations executed (init + optimization).
   * - ``n_init``
     - ``int``
     - Number of initialization iterations.
   * - ``n_optimization``
     - ``int``
     - Number of optimization iterations (after init phase).


Convergence
^^^^^^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``convergence_data``
     - ``list[float]``
     - Best score at each iteration. Monotonically non-decreasing.
   * - ``n_score_improvements``
     - ``int``
     - Number of times the best score improved.
   * - ``last_improvement``
     - ``int``
     - Iteration index of the last score improvement.
   * - ``longest_plateau``
     - ``tuple[int, int, int]``
     - ``(length, start_iter, end_iter)`` of the longest stretch without improvement.


Acceptance
^^^^^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``acceptance_rate``
     - ``float``
     - Fraction of proposed positions that were accepted (0.0 to 1.0).
   * - ``n_accepted``
     - ``int``
     - Number of accepted position changes.
   * - ``n_proposed``
     - ``int``
     - Number of proposed positions.


Score Statistics
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``score_min``
     - ``float``
     - Minimum score (excludes ``inf``/``nan``).
   * - ``score_max``
     - ``float``
     - Maximum score (excludes ``inf``/``nan``).
   * - ``score_mean``
     - ``float``
     - Mean of all scores (excludes ``inf``/``nan``).
   * - ``score_std``
     - ``float``
     - Standard deviation of scores (excludes ``inf``/``nan``).
   * - ``n_invalid``
     - ``int``
     - Number of evaluations that returned ``inf`` or ``nan``.


Timing
^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``total_time``
     - ``float``
     - Total wall time in seconds.
   * - ``eval_time``
     - ``float``
     - Time spent in the objective function.
   * - ``overhead_time``
     - ``float``
     - Time spent in optimizer logic (total - eval).
   * - ``eval_pct``
     - ``float``
     - Evaluation time as percentage of total.
   * - ``overhead_pct``
     - ``float``
     - Optimizer overhead as percentage of total.
   * - ``avg_eval_time``
     - ``float``
     - Average time per objective function call.
   * - ``avg_iter_time``
     - ``float``
     - Average time per iteration (including overhead).
   * - ``throughput``
     - ``float``
     - Iterations per second.


All Results
^^^^^^^^^^^

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``results``
     - ``list[dict]``
     - All evaluated positions as list of dicts. Each dict contains ``score``
       and all parameter values.

----


``opt.data.raw`` Properties
---------------------------

The ``raw`` sub-accessor provides direct access to internal tracking lists.
These are the underlying data structures, returned by reference (no copy).
Use ``opt.data`` properties for computed metrics, and ``opt.data.raw`` when
you need the raw lists for custom analysis.


.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``scores_proposed``
     - ``list[float]``
     - All scores in order of proposal (one per iteration).
   * - ``scores_accepted``
     - ``list[float]``
     - Scores at each accepted (current) position change.
   * - ``scores_best``
     - ``list[float]``
     - Score at each best-position update.
   * - ``scores_all``
     - ``list[float]``
     - All scores in evaluation order.
   * - ``positions_proposed``
     - ``list[dict]``
     - All proposed positions as parameter dicts. Computed on access.
   * - ``positions_accepted``
     - ``list[dict]``
     - Accepted positions as parameter dicts. Computed on access.
   * - ``eval_times``
     - ``list[float]``
     - Time spent in objective function per evaluation.
   * - ``iter_times``
     - ``list[float]``
     - Total time per iteration including optimizer overhead.
   * - ``convergence``
     - ``list[float]``
     - Best score seen at each iteration.
   * - ``improvement_iterations``
     - ``list[int]``
     - Iteration indices where the best score improved.


.. tip::

   **When to use raw vs data:**

   - Use ``opt.data.best_score`` to get a single computed value.
   - Use ``opt.data.raw.scores_proposed`` when you need the full list for
     custom analysis, plotting, or export.

----


Examples
--------

Convergence Plot
^^^^^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as plt

    opt.search(objective, n_iter=200, verbosity=False)

    plt.plot(opt.data.convergence_data)
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.title("Convergence Curve")
    plt.show()


Checking if n_iter is Too High
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    opt.search(objective, n_iter=1000, verbosity=False)

    d = opt.data
    print(f"Last improvement at iter {d.last_improvement} of {d.n_iter}")
    print(f"Longest plateau: {d.longest_plateau[0]} iterations")

    if d.last_improvement < d.n_iter * 0.2:
        print("Consider reducing n_iter or trying a different optimizer.")


Comparing Optimizers
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from gradient_free_optimizers import (
        HillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        BayesianOptimizer,
    )

    results = {}
    for Opt in [HillClimbingOptimizer, SimulatedAnnealingOptimizer, BayesianOptimizer]:
        opt = Opt(search_space, random_state=42)
        opt.search(objective, n_iter=100, verbosity=False)
        d = opt.data
        results[Opt.__name__] = {
            "score": d.best_score,
            "improvements": d.n_score_improvements,
            "acceptance": f"{d.acceptance_rate:.1%}",
            "eval_time": f"{d.eval_time:.3f}s",
        }

    for name, metrics in results.items():
        print(f"{name}: {metrics}")
