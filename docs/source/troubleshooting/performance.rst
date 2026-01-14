.. _troubleshooting_performance:

====================
Performance Problems
====================

Solutions for slow optimization and high memory usage.

----

Slow Optimization
=================

Profile the Objective Function
-------------------------------

**Problem**: Optimization takes too long.

**First step**: Determine if the bottleneck is your objective function or the optimizer.

.. code-block:: python

    import time
    from gradient_free_optimizers import RandomSearchOptimizer

    def objective(params):
        start = time.time()
        result = your_expensive_function(params)
        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"Slow evaluation: {elapsed:.2f}s")
        return result

    # Time total optimization
    start = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective, n_iter=100)
    total = time.time() - start

    print(f"Total time: {total:.2f}s")
    print(f"Time per iteration: {total/100:.2f}s")


If per-iteration time is high, your objective function is the bottleneck (most common).
If it's low, the optimizer overhead is the issue.

----

Objective Function Optimization
================================

Use Caching
-----------

**Problem**: Re-evaluating same parameters multiple times.

**Solution**: Enable GFO's built-in memory:

.. code-block:: python

    opt.search(objective, n_iter=1000, memory=True)  # Default

Or implement custom caching:

.. code-block:: python

    from functools import lru_cache

    @lru_cache(maxsize=1000)
    def cached_objective(x, y):
        return expensive_computation(x, y)

    def objective(params):
        return cached_objective(params["x"], params["y"])


Vectorize Computations
-----------------------

**Problem**: Looping in NumPy when vectorization is possible.

**Slow**:

.. code-block:: python

    def objective(params):
        result = 0
        for i in range(len(data)):
            result += compute(data[i], params["x"])
        return result

**Fast**:

.. code-block:: python

    def objective(params):
        return np.sum(vectorized_compute(data, params["x"]))


Use Numba or Cython
-------------------

**Problem**: Pure Python loops are slow.

**Solution**: Compile with Numba:

.. code-block:: python

    from numba import jit

    @jit(nopython=True)
    def fast_computation(x, y):
        result = 0.0
        for i in range(1000000):
            result += x * y + i
        return result

    def objective(params):
        return fast_computation(params["x"], params["y"])


Reduce Model Complexity
------------------------

**Problem**: ML model training too slow.

**Solutions**:

1. Use surrogate models during optimization:

   .. code-block:: python

       from sklearn.datasets import make_classification
       from sklearn.ensemble import RandomForestClassifier
       from sklearn.model_selection import cross_val_score

       X, y = make_classification(n_samples=1000)  # Subset

       def objective(params):
           model = RandomForestClassifier(
               n_estimators=params["n_estimators"],
               max_depth=params["max_depth"],
           )
           # Use 3-fold instead of 10-fold
           return cross_val_score(model, X, y, cv=3).mean()

2. Early stopping:

   .. code-block:: python

       opt.search(
           objective,
           n_iter=1000,
           early_stopping={"n_iter_no_change": 50}  # Stop if stuck
       )

----

Optimizer-Level Optimization
=============================

Choose a Faster Algorithm
--------------------------

**Problem**: Some algorithms have high overhead.

**Algorithm Speed Comparison** (fastest to slowest):

1. **RandomSearchOptimizer** - No overhead
2. **GridSearchOptimizer** - Minimal overhead
3. **HillClimbingOptimizer** - Very low overhead
4. **ParticleSwarmOptimizer** - Low overhead
5. **BayesianOptimizer** - High overhead (GP training is O(n³))

For cheap objective functions (< 0.01s), use simple algorithms:

.. code-block:: python

    # Fast objective? Use fast optimizer
    from gradient_free_optimizers import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_space)
    opt.search(fast_objective, n_iter=10000)  # No problem

For expensive objectives, Bayesian Optimization is worth the overhead:

.. code-block:: python

    # Slow objective? SMBO algorithms help
    from gradient_free_optimizers import BayesianOptimizer

    opt = BayesianOptimizer(search_space)
    opt.search(expensive_objective, n_iter=100)  # Fewer iterations needed


Scale SMBO for Many Iterations
-------------------------------

**Problem**: Bayesian Optimization slows down with many iterations.

**Solution**: Use ForestOptimizer instead:

.. code-block:: python

    # For 100+ iterations
    from gradient_free_optimizers import ForestOptimizer

    opt = ForestOptimizer(search_space)
    opt.search(objective, n_iter=500)  # Scales better than Bayesian


Reduce Search Space Granularity
--------------------------------

**Problem**: Too many points in search space.

**Solution**: Use coarser discretization:

.. code-block:: python

    # Fine grid (slow)
    search_space = {
        "x": np.linspace(0, 1, 1000),  # 1000 points
        "y": np.linspace(0, 1, 1000),  # 1000 points
    }
    # Total: 1,000,000 combinations

    # Coarse grid (faster)
    search_space = {
        "x": np.linspace(0, 1, 50),  # 50 points
        "y": np.linspace(0, 1, 50),  # 50 points
    }
    # Total: 2,500 combinations

For continuous optimization, this is usually sufficient.


Disable Progress Bars
----------------------

**Problem**: Progress bar rendering has overhead.

**Solution**: Disable verbosity:

.. code-block:: python

    opt.search(objective, n_iter=1000, verbosity=[])  # Silent


----

High Memory Usage
=================

Disable Search Data Collection
-------------------------------

**Problem**: ``search_data`` DataFrame grows large.

**Solution**: Disable memory if you don't need search history:

.. code-block:: python

    opt.search(objective, n_iter=10000, memory=False)


Clear Data Periodically
------------------------

**Problem**: Long-running optimization accumulates data.

**Solution**: Use ask-tell with manual clearing:

.. code-block:: python

    opt = HillClimbingOptimizer(search_space)
    opt.setup_search(objective, n_iter=10000)

    for i in range(10000):
        params = opt.ask()
        score = objective(params)
        opt.tell(params, score)

        # Clear every 1000 iterations
        if i % 1000 == 0:
            opt.search_data = opt.search_data.iloc[-100:]  # Keep last 100


Reduce Surrogate Model Size
----------------------------

**Problem**: SMBO algorithms store many training points.

**Solution**: Limit training data:

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer

    # Use ForestOptimizer for better memory scaling
    from gradient_free_optimizers import ForestOptimizer
    opt = ForestOptimizer(search_space)


----

Parallel Optimization
=====================

GFO Doesn't Support Parallelism Directly
-----------------------------------------

**Problem**: Want to use multiple CPU cores.

**Solution**: Use Hyperactive (built on GFO) for parallel optimization:

.. code-block:: python

    # Install Hyperactive
    # pip install hyperactive

    from hyperactive.opt import HillClimbing

    optimizer = HillClimbing(
        search_space,
        n_iter=100,
        experiment=objective,
        n_jobs=4  # Use 4 cores
    )
    best = optimizer.solve()

Or manually parallelize with multiprocessing:

.. code-block:: python

    from multiprocessing import Pool
    from gradient_free_optimizers import RandomSearchOptimizer

    def run_optimization(seed):
        opt = RandomSearchOptimizer(search_space, random_state=seed)
        opt.search(objective, n_iter=100)
        return opt.best_score, opt.best_para

    with Pool(4) as pool:
        results = pool.map(run_optimization, range(4))

    # Find best across all runs
    best_score, best_params = max(results, key=lambda x: x[0])


----

Benchmarking
============

Measure True Performance
-------------------------

.. code-block:: python

    import time
    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def benchmark_optimizer(optimizer_class, n_runs=10):
        times = []
        scores = []

        for i in range(n_runs):
            opt = optimizer_class(search_space, random_state=i)

            start = time.time()
            opt.search(objective, n_iter=100)
            elapsed = time.time() - start

            times.append(elapsed)
            scores.append(opt.best_score)

        print(f"{optimizer_class.__name__}")
        print(f"  Avg time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
        print(f"  Avg score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Compare optimizers
    from gradient_free_optimizers import (
        RandomSearchOptimizer,
        HillClimbingOptimizer,
        BayesianOptimizer,
    )

    benchmark_optimizer(RandomSearchOptimizer)
    benchmark_optimizer(HillClimbingOptimizer)
    benchmark_optimizer(BayesianOptimizer)


----

Still Too Slow?
===============

If performance is still an issue:

1. **Profile your objective function**: Use ``cProfile`` or ``line_profiler``
2. **Reduce problem size**: Smaller search space, fewer parameters
3. **Try different algorithms**: Some are faster for your specific problem
4. **Consider GPU acceleration**: For ML models, use GPU training
5. **Use Hyperactive**: For parallel optimization across cores

See :ref:`troubleshooting_help` for more assistance.
