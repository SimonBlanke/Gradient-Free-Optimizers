:orphan:

.. _search_data:

=====================
Search Data & Summary
=====================

After running ``search()``, GFO provides two ways to understand what happened:
a **CLI summary** printed directly to the terminal, and a programmatic
**DataFrame** of all evaluations via ``opt.search_data``.


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

    # Access results programmatically
    print(opt.best_score)   # float
    print(opt.best_para)    # dict
    print(opt.search_data)  # pandas DataFrame of all evaluations

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

This prints a compact reference of all summary fields with their descriptions.

----


``opt.search_data`` DataFrame
-----------------------------

Per-iteration results are available as a pandas DataFrame via the flat
attribute ``opt.search_data``. Each row is one evaluation; columns include
``score``, all parameter values, and any custom metrics returned by the
objective.

.. code-block:: python

    opt.search(objective, n_iter=200, verbosity=False)
    df = opt.search_data
    print(df.head())
    print(df["score"].describe())


Examples
--------

Convergence Plot
^^^^^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as plt

    opt.search(objective, n_iter=200, verbosity=False)

    df = opt.search_data
    plt.plot(df["score"].cummax())
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.title("Convergence Curve")
    plt.show()


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
        results[Opt.__name__] = opt.best_score

    for name, score in results.items():
        print(f"{name}: {score}")
