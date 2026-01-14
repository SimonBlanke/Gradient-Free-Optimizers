.. _troubleshooting_runtime:

==============
Runtime Errors
==============

Solutions for common errors during optimization.

----

TypeError
=========

"objective_function() got an unexpected keyword argument"
----------------------------------------------------------

**Problem**: Objective function doesn't accept a dictionary parameter.

**Wrong**:

.. code-block:: python

    def objective(x, y):  # Wrong signature
        return -(x**2 + y**2)

**Correct**:

.. code-block:: python

    def objective(params):  # Correct: takes dict
        return -(params["x"]**2 + params["y"]**2)


"unhashable type: 'dict'"
--------------------------

**Problem**: Trying to use mutable objects (dicts, lists) in search space.

**Wrong**:

.. code-block:: python

    search_space = {
        "config": [{"a": 1}, {"b": 2}]  # Dicts not hashable
    }

**Correct**:

.. code-block:: python

    search_space = {
        "config": np.array(["config_a", "config_b"])  # Use strings
    }

Handle configs in your objective function:

.. code-block:: python

    configs = {"config_a": {"a": 1}, "config_b": {"b": 2}}

    def objective(params):
        config = configs[params["config"]]
        # Use config...


"'numpy.ndarray' object is not callable"
-----------------------------------------

**Problem**: Passing search space arrays instead of the objective function.

**Wrong**:

.. code-block:: python

    opt.search(search_space, n_iter=100)  # Wrong order

**Correct**:

.. code-block:: python

    opt.search(objective_function, n_iter=100)  # Correct order


----

ValueError
==========

"could not broadcast input array"
----------------------------------

**Problem**: Inconsistent array shapes in search space.

**Solution**: Ensure all parameter arrays are 1D NumPy arrays:

.. code-block:: python

    # Wrong
    search_space = {
        "x": [[1, 2], [3, 4]],  # 2D array
    }

    # Correct
    search_space = {
        "x": np.array([1, 2, 3, 4]),  # 1D array
    }


"all the input arrays must have same number of dimensions"
-----------------------------------------------------------

**Problem**: Mixing lists and arrays in search space.

**Solution**: Convert everything to NumPy arrays:

.. code-block:: python

    import numpy as np

    search_space = {
        "x": np.array([1, 2, 3]),      # NumPy array
        "y": np.linspace(0, 10, 50),   # NumPy array
        "z": np.arange(1, 100),        # NumPy array
    }


"setting an array element with a sequence"
-------------------------------------------

**Problem**: Search space contains nested structures.

**Solution**: Flatten or use proper array construction:

.. code-block:: python

    # If you have nested data, flatten it
    values = np.array([item for sublist in nested_list for item in sublist])

    search_space = {"param": values}


----

KeyError
========

"'x' not found in parameters"
------------------------------

**Problem**: Objective function references a parameter not in search space.

**Solution**: Match parameter names exactly:

.. code-block:: python

    search_space = {
        "learning_rate": np.logspace(-4, -1, 20),
        "n_estimators": np.arange(10, 200, 10),
    }

    def objective(params):
        # Use exact names from search_space
        lr = params["learning_rate"]   # Not params["lr"]
        n = params["n_estimators"]     # Not params["n"]
        return score


----

AttributeError
==============

"'OptimizerClass' object has no attribute 'best_para'"
-------------------------------------------------------

**Problem**: Accessing results before running optimization.

**Solution**: Call ``search()`` first:

.. code-block:: python

    opt = HillClimbingOptimizer(search_space)

    # This fails - no search yet
    # print(opt.best_para)

    # Run search first
    opt.search(objective, n_iter=100)

    # Now results are available
    print(opt.best_para)
    print(opt.best_score)


"module 'gradient_free_optimizers' has no attribute 'OptimizerName'"
---------------------------------------------------------------------

**Problem**: Typo in optimizer name or using old API.

**Solution**: Check spelling and use current API:

.. code-block:: python

    # Common mistakes
    # from gradient_free_optimizers import BayesOptimizer  # Wrong
    # from gradient_free_optimizers import HillClimbing     # Wrong

    # Correct
    from gradient_free_optimizers import BayesianOptimizer
    from gradient_free_optimizers import HillClimbingOptimizer


----

Memory Errors
=============

"MemoryError" or System Crashes
--------------------------------

**Problem**: Search space or memory cache too large.

**Solutions**:

1. Reduce search space granularity:

   .. code-block:: python

       # Instead of 10000 points
       # "x": np.linspace(-10, 10, 10000)

       # Use fewer points
       "x": np.linspace(-10, 10, 100)

2. Disable memory caching:

   .. code-block:: python

       opt.search(objective, n_iter=100, memory=False)

3. For SMBO, use simpler surrogate:

   .. code-block:: python

       # Instead of BayesianOptimizer (expensive for many iterations)
       # Use ForestOptimizer (more scalable)
       from gradient_free_optimizers import ForestOptimizer


----

Constraint Errors
=================

Optimization Stuck or Very Slow
--------------------------------

**Problem**: Constraints too restrictive, no valid positions found.

**Solution**: Verify constraints allow valid solutions:

.. code-block:: python

    def constraint(params):
        # Check this actually allows some solutions
        return params["x"] + params["y"] < 100

    # Test the constraint
    search_space = {"x": np.arange(0, 10), "y": np.arange(0, 10)}

    # Try a few combinations
    test_params = [
        {"x": 0, "y": 0},
        {"x": 5, "y": 5},
        {"x": 9, "y": 9},
    ]

    for p in test_params:
        print(f"{p}: {constraint(p)}")  # Should see some True


"Constraint function returned non-boolean"
-------------------------------------------

**Problem**: Constraint must return True/False, not numbers.

**Wrong**:

.. code-block:: python

    def constraint(params):
        return params["x"] + params["y"]  # Returns number

**Correct**:

.. code-block:: python

    def constraint(params):
        return params["x"] + params["y"] < 10  # Returns bool


----

SMBO-Specific Errors
====================

"not enough values to unpack"
------------------------------

**Problem**: Not enough initial evaluations for surrogate model.

**Solution**: Increase initialization or n_iter:

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer

    opt = BayesianOptimizer(
        search_space,
        initialize={"random": 10}  # At least 10 initial points
    )
    opt.search(objective, n_iter=50)  # Sufficient iterations


"Singular matrix" in Bayesian Optimization
-------------------------------------------

**Problem**: Duplicate evaluations or ill-conditioned covariance matrix.

**Solutions**:

1. Ensure search space has variation:

   .. code-block:: python

       # Not just 2-3 points
       search_space = {"x": np.linspace(-5, 5, 50)}

2. Add noise to avoid exact duplicates:

   .. code-block:: python

       def objective(params):
           result = expensive_function(params)
           # Add tiny noise if needed
           return result + np.random.normal(0, 1e-6)


----

Debug Tips
==========

Enable Verbose Output
---------------------

.. code-block:: python

    opt.search(
        objective,
        n_iter=100,
        verbosity=["progress_bar", "print_results", "print_times"]
    )


Check Intermediate Values
--------------------------

.. code-block:: python

    def objective(params):
        print(f"Evaluating: {params}")
        result = compute_score(params)
        print(f"Score: {result}")
        return result


Inspect Search Data
-------------------

.. code-block:: python

    opt.search(objective, n_iter=100)

    # View all evaluations
    print(opt.search_data)

    # Check for patterns
    print(opt.search_data.describe())
    print(opt.search_data.isnull().sum())


----

Still Getting Errors?
=====================

If the error persists:

1. **Simplify**: Create a minimal example that reproduces the error
2. **Check versions**: Ensure GFO and dependencies are up to date
3. **Report**: See :ref:`troubleshooting_help` for how to report issues

Include in your report:

- Full error traceback
- GFO version
- Minimal code that reproduces the error
- Python and NumPy versions
