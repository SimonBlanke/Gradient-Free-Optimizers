======================
The search() Method
======================

The ``search()`` method is the primary interface for running optimization in
GFO. Once you have defined a search space and created an optimizer, calling
``search()`` starts the optimization loop. The method accepts your objective
function, the number of iterations, and optional parameters for stopping
conditions, memory caching, and output control. After the search completes,
results are available as attributes on the optimizer object.


.. grid:: 1

   .. grid-item-card::
      :class-card: sd-border-primary gfo-compact

      .. code-block:: python

          opt = Optimizer(search_space)        # 1. Create
          opt.search(objective, n_iter=100)    # 2. Search
          opt.best_para                        # 3. Results


Basic Usage
-----------

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=100)

    print(opt.best_para)   # Best parameters
    print(opt.best_score)  # Best score


Full Signature
--------------

.. code-block:: python

    opt.search(
        objective_function,           # Required: function to optimize
        n_iter,                        # Required: number of iterations

        # Stopping conditions
        max_time=None,                 # Max wall-clock seconds
        max_score=None,                # Stop when score >= this value
        early_stopping=None,           # Early stopping config

        # Memory
        memory=True,                   # Cache evaluations
        memory_warm_start=None,        # DataFrame from previous run

        # Output
        verbosity=["progress_bar", "print_results"],

        # Direction
        optimum="maximum",             # "maximum" or "minimum"
    )


Parameters
----------

**objective_function**

A callable that takes a parameter dictionary and returns a score:

.. code-block:: python

    def objective(params):
        return -(params["x"]**2 + params["y"]**2)

**n_iter**

Number of iterations (function evaluations):

.. code-block:: python

    opt.search(objective, n_iter=500)

**max_time**

Maximum wall-clock time in seconds:

.. code-block:: python

    opt.search(objective, n_iter=10000, max_time=3600)

**max_score**

Stop early if this score is reached:

.. code-block:: python

    opt.search(objective, n_iter=1000, max_score=0.99)

**early_stopping**

Stop if no improvement for N iterations:

.. code-block:: python

    opt.search(objective, n_iter=1000, early_stopping={"n_iter_no_change": 50})

**memory**

Enable/disable evaluation caching:

.. code-block:: python

    opt.search(objective, n_iter=500, memory=True)

**memory_warm_start**

DataFrame with previous evaluations:

.. code-block:: python

    opt.search(objective, n_iter=500, memory_warm_start=previous_data)

**verbosity**

Control output:

.. code-block:: python

    opt.search(objective, n_iter=100, verbosity=[])  # Silent
    opt.search(objective, n_iter=100, verbosity=["progress_bar"])
    opt.search(objective, n_iter=100, verbosity=["print_results"])
    opt.search(objective, n_iter=100, verbosity=["progress_bar", "print_results"])

**optimum**

Optimization direction:

.. code-block:: python

    opt.search(objective, n_iter=100, optimum="maximum")  # Default
    opt.search(objective, n_iter=100, optimum="minimum")


Result Attributes
-----------------

After search completes:

.. code-block:: python

    opt.best_para    # dict: Best parameters found
    opt.best_score   # float: Best score achieved
    opt.search_data  # DataFrame: All evaluations


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer

    def objective(params):
        return -(params["x"]**2 + params["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = BayesianOptimizer(search_space)
    opt.search(
        objective,
        n_iter=50,
        max_time=60,
        early_stopping={"n_iter_no_change": 20},
        memory=True,
        verbosity=["progress_bar", "print_results"],
    )

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")
    print(f"Evaluations: {len(opt.search_data)}")
