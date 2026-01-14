===================
Stopping Conditions
===================

GFO supports multiple stopping conditions that can be combined.


Available Conditions
--------------------

**Iteration count (n_iter):**

.. code-block:: python

    opt.search(objective, n_iter=1000)

**Time limit (max_time):**

.. code-block:: python

    opt.search(objective, n_iter=10000, max_time=3600)  # 1 hour max

**Target score (max_score):**

.. code-block:: python

    opt.search(objective, n_iter=1000, max_score=0.99)

**Early stopping:**

.. code-block:: python

    opt.search(
        objective,
        n_iter=1000,
        early_stopping={"n_iter_no_change": 50}
    )


Combining Conditions
--------------------

Use multiple conditions together. Optimization stops when ANY condition is met:

.. code-block:: python

    opt.search(
        objective,
        n_iter=5000,              # Max 5000 iterations
        max_time=1800,            # OR max 30 minutes
        max_score=0.999,          # OR target reached
        early_stopping={
            "n_iter_no_change": 100  # OR no improvement for 100 iterations
        }
    )


Early Stopping Details
----------------------

Early stopping monitors the best score and stops if no improvement is found
for a specified number of iterations:

.. code-block:: python

    early_stopping = {
        "n_iter_no_change": 50,  # Stop after 50 iterations without improvement
    }

This is useful when:

- You don't know how many iterations are needed
- The function converges before the iteration budget
- You want to save computation time


Examples
--------

**ML hyperparameter tuning:**

.. code-block:: python

    opt.search(
        objective,
        n_iter=200,
        max_time=7200,            # 2 hour limit
        max_score=0.98,           # Stop if 98% accuracy reached
        early_stopping={"n_iter_no_change": 30},
    )

**Quick exploration:**

.. code-block:: python

    opt.search(
        objective,
        n_iter=100,
        max_time=60,  # Quick 1-minute exploration
    )

**Long-running optimization:**

.. code-block:: python

    opt.search(
        objective,
        n_iter=10000,
        max_time=86400,  # 24-hour limit
        early_stopping={"n_iter_no_change": 500},
    )


Checking Why Optimization Stopped
---------------------------------

After optimization, you can check how many iterations were completed:

.. code-block:: python

    opt.search(objective, n_iter=1000, max_score=0.99)

    print(f"Completed iterations: {len(opt.search_data)}")
    print(f"Best score: {opt.best_score}")

    if len(opt.search_data) < 1000:
        if opt.best_score >= 0.99:
            print("Stopped: target score reached")
        else:
            print("Stopped: early stopping or time limit")
