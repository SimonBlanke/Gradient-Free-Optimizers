================
Memory & Caching
================

GFO can cache function evaluations to avoid redundant computation and support
warm-starting from previous optimization runs.


.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Memory
      :class-card: sd-border-primary gfo-compact

      ``memory=True``

      Cache evaluations within a single run.
      Avoids redundant function calls.

   .. grid-item-card:: Warm Start
      :class-card: sd-border-success gfo-compact

      ``memory_warm_start=df``

      Continue from a previous optimization
      run using saved results.


Memory Caching
--------------

Enable memory to cache evaluations during search:

.. code-block:: python

    opt.search(objective, n_iter=500, memory=True)

When the optimizer proposes a position it has already evaluated, the cached
score is returned instead of calling the objective function again.

**Benefits:**

- Avoids redundant evaluations of expensive functions
- Enables effective local search in discrete spaces
- Reduces total computation time


Warm Start
----------

Continue optimization from previous results:

.. code-block:: python

    # First run
    opt1 = BayesianOptimizer(search_space)
    opt1.search(objective, n_iter=25)
    previous_data = opt1.search_data

    # Second run, starting from first run's results
    opt2 = BayesianOptimizer(search_space)
    opt2.search(
        objective,
        n_iter=25,
        memory_warm_start=previous_data,
    )

The optimizer uses the warm start data to:

- Avoid re-evaluating known positions
- Inform surrogate models (for SMBO algorithms)
- Start from the best known position


Saving and Loading Results
--------------------------

.. code-block:: python

    import pandas as pd

    # Save after optimization
    opt.search_data.to_csv("results.csv", index=False)

    # Load for warm start
    previous_data = pd.read_csv("results.csv")
    opt.search(objective, memory_warm_start=previous_data)


Best Practices
--------------

1. **Always enable memory for expensive functions**: The overhead is minimal
2. **Save results periodically**: Allows recovery from interruptions
3. **Use warm start for iterative optimization**: Refine results over multiple sessions
4. **Combine different algorithms**: Run random search first, then warm-start Bayesian


When to Disable Memory
----------------------

Memory may not help when:

- The search space is very large (low chance of revisiting)
- The objective function is very cheap
- You want independent random samples

.. code-block:: python

    opt.search(objective, n_iter=500, memory=False)
