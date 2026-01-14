.. _troubleshooting_results:

===============
Result Problems
===============

Solutions for unexpected optimization results.

----

Wrong Optimization Direction
=============================

Minimizing Instead of Maximizing
---------------------------------

**Problem**: GFO maximizes by default, but you want to minimize.

**Solution**: Negate your objective function:

.. code-block:: python

    # Want to minimize this
    def loss_function(params):
        return mean_squared_error(y_true, y_pred)

    # Negate for maximization
    def objective(params):
        return -loss_function(params)

    opt.search(objective, n_iter=100)

    # Best score is negative of best loss
    best_loss = -opt.best_score


Or use ``optimum="minimum"`` parameter:

.. code-block:: python

    opt.search(
        loss_function,
        n_iter=100,
        optimum="minimum"  # Minimize instead of maximize
    )


Maximizing Instead of Minimizing
---------------------------------

**Problem**: Accuracy/score is going down instead of up.

**Solution**: Check you're not negating when you shouldn't:

.. code-block:: python

    # For accuracy (higher is better), no negation needed
    def objective(params):
        return model_accuracy(params)  # Correct

    # Not this
    # return -model_accuracy(params)  # Wrong!


----

Stuck in Local Optima
=====================

Not Exploring Enough
--------------------

**Problem**: Optimizer finds a local optimum and stops improving.

**Solutions**:

1. Use algorithms with better exploration:

   .. code-block:: python

       # Instead of HillClimbingOptimizer (exploits)
       from gradient_free_optimizers import SimulatedAnnealingOptimizer

       opt = SimulatedAnnealingOptimizer(
           search_space,
           start_temp=10.0,  # High temperature for more exploration
           annealing_rate=0.99,
       )

2. Add random restarts:

   .. code-block:: python

       opt = HillClimbingOptimizer(
           search_space,
           rand_rest_p=0.1,  # 10% chance of random restart each iteration
       )

3. Use population-based methods:

   .. code-block:: python

       from gradient_free_optimizers import ParticleSwarmOptimizer

       opt = ParticleSwarmOptimizer(
           search_space,
           population=20,  # Multiple agents explore simultaneously
       )

4. Run multiple times with different seeds:

   .. code-block:: python

       best_results = []
       for seed in range(10):
           opt = BayesianOptimizer(search_space, random_state=seed)
           opt.search(objective, n_iter=50)
           best_results.append((opt.best_score, opt.best_para))

       # Get overall best
       best_score, best_params = max(best_results, key=lambda x: x[0])


Not Enough Iterations
---------------------

**Problem**: Optimization stops before finding good solution.

**Solution**: Increase iterations:

.. code-block:: python

    # Too few
    opt.search(objective, n_iter=10)  # Probably insufficient

    # More reasonable
    opt.search(objective, n_iter=100)  # Better

    # For complex problems
    opt.search(objective, n_iter=1000)  # Even better

Rule of thumb: At least 50-100 iterations per parameter dimension.


----

Unrealistic Best Score
======================

Score is NaN or Inf
-------------------

**Problem**: Objective function returns invalid values.

**Solution**: Add error handling:

.. code-block:: python

    import numpy as np

    def objective(params):
        try:
            result = compute_score(params)

            # Check for invalid values
            if np.isnan(result) or np.isinf(result):
                return -1e10  # Return very bad score

            return result
        except Exception as e:
            print(f"Error with {params}: {e}")
            return -1e10  # Return very bad score on error


Score Outside Expected Range
-----------------------------

**Problem**: Getting scores that don't make sense for your problem.

**Solution**: Verify objective function:

.. code-block:: python

    def objective(params):
        # Add debugging
        print(f"Params: {params}")

        score = compute_score(params)

        print(f"Score: {score}")

        # Add sanity checks
        assert 0 <= score <= 1, f"Score out of range: {score}"

        return score


Score Not Changing
------------------

**Problem**: All evaluations return the same score.

**Solution**: Check that parameters actually affect the result:

.. code-block:: python

    def objective(params):
        print(f"Evaluating: {params}")

        # Make sure you USE the parameters!
        result = model.train(
            learning_rate=params["learning_rate"],  # Use these!
            n_estimators=params["n_estimators"],
        )

        # Not this (parameters ignored):
        # result = model.train()  # Uses default params!

        return result


----

Non-Reproducible Results
=========================

Different Results Each Run
---------------------------

**Problem**: Running the same code gives different results.

**Solution**: Set random seed:

.. code-block:: python

    opt = HillClimbingOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=100)

Also set seeds for libraries used in objective:

.. code-block:: python

    import numpy as np
    import random

    # Set all seeds
    random.seed(42)
    np.random.seed(42)

    # For ML libraries
    import torch
    torch.manual_seed(42)

    # For sklearn
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)


----

Poor Quality Solutions
======================

Best Parameters Don't Work Well
--------------------------------

**Problem**: Best parameters from optimization perform poorly in practice.

**Possible causes**:

1. **Overfitting to evaluation data**:

   .. code-block:: python

       # Use train/val/test splits properly
       from sklearn.model_selection import train_test_split

       X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
       X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

       def objective(params):
           model.fit(X_train, y_train)
           # Evaluate on validation set (not train!)
           return model.score(X_val, y_val)

       # After optimization, test on test set
       final_model.fit(X_train, y_train)
       test_score = final_model.score(X_test, y_test)

2. **Wrong search space bounds**:

   .. code-block:: python

       # Check your bounds make sense
       search_space = {
           "learning_rate": np.logspace(-5, 0, 50),  # 0.00001 to 1.0
           "n_estimators": np.arange(10, 500, 10),   # 10 to 490
       }

       # Not this (too narrow)
       # "learning_rate": np.logspace(-3, -2, 10),  # Only 0.001 to 0.01

3. **Noisy objective function**:

   .. code-block:: python

       # Average multiple evaluations
       def objective(params):
           scores = []
           for _ in range(5):  # 5 runs
               score = evaluate_with_different_seed(params)
               scores.append(score)
           return np.mean(scores)


Search Space Doesn't Contain Optimum
-------------------------------------

**Problem**: Best result is at boundary of search space.

**Solution**: Expand search space bounds:

.. code-block:: python

    # Check if best is at boundary
    opt.search(objective, n_iter=100)

    for param_name, param_value in opt.best_para.items():
        space_array = search_space[param_name]
        if param_value == space_array[0]:
            print(f"{param_name} at lower bound!")
        if param_value == space_array[-1]:
            print(f"{param_name} at upper bound!")

    # Expand bounds and re-run
    expanded_space = {
        "x": np.linspace(-20, 20, 100),  # Was -10 to 10
        "y": np.linspace(-20, 20, 100),
    }


----

Comparing Results
=================

Comparing Different Algorithms
-------------------------------

**Problem**: Want to know which optimizer performs best.

**Solution**: Run systematic comparison:

.. code-block:: python

    from gradient_free_optimizers import (
        RandomSearchOptimizer,
        HillClimbingOptimizer,
        ParticleSwarmOptimizer,
        BayesianOptimizer,
    )

    optimizers = [
        RandomSearchOptimizer,
        HillClimbingOptimizer,
        ParticleSwarmOptimizer,
        BayesianOptimizer,
    ]

    results = {}
    for opt_class in optimizers:
        scores = []
        for seed in range(10):  # 10 runs each
            opt = opt_class(search_space, random_state=seed)
            opt.search(objective, n_iter=100)
            scores.append(opt.best_score)

        results[opt_class.__name__] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "best": max(scores),
        }

    # Print comparison
    for name, stats in results.items():
        print(f"{name:30s}: {stats['mean']:.4f} ± {stats['std']:.4f}")


----

Validation
==========

Sanity Checks
-------------

Always validate your optimization results:

.. code-block:: python

    # 1. Run a simple baseline
    random_opt = RandomSearchOptimizer(search_space)
    random_opt.search(objective, n_iter=10)
    baseline = random_opt.best_score

    # 2. Run your optimizer
    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=100)

    # 3. Compare
    improvement = opt.best_score - baseline
    print(f"Baseline: {baseline:.4f}")
    print(f"Optimized: {opt.best_score:.4f}")
    print(f"Improvement: {improvement:.4f}")

    # Should see improvement!
    assert opt.best_score > baseline, "Optimization didn't beat random search!"


Manual Verification
-------------------

.. code-block:: python

    # Test the best parameters manually
    best_params = opt.best_para
    manual_score = objective(best_params)

    print(f"Reported best score: {opt.best_score}")
    print(f"Manual evaluation: {manual_score}")

    # Should match (within noise)
    assert abs(manual_score - opt.best_score) < 0.01, "Results don't match!"


----

Still Having Issues?
====================

If results still look wrong:

1. **Simplify**: Test on a simple known function (sphere, Rosenbrock)
2. **Debug**: Add print statements to track what's happening
3. **Visualize**: Plot the search data to see the optimization trajectory
4. **Ask for help**: See :ref:`troubleshooting_help`

When reporting:

- Your objective function (simplified version)
- Search space definition
- Expected vs. actual results
- Algorithm and parameters used
