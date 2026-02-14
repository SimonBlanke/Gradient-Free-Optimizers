===================
Objective Functions
===================

The objective function is the core of every optimization run. The optimizer
proposes parameter combinations, your objective function evaluates them, and
the optimizer learns from the results to propose better combinations over
time. An objective function takes a dictionary of parameters as input and
returns a single numeric score. By default GFO maximizes this score, meaning
higher values are considered better.


.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Input
      :class-card: sd-border-primary gfo-compact

      ``params = {"x": 0.5, "y": 1.2}``

      A dictionary of parameter values
      proposed by the optimizer.

   .. grid-item-card:: Output
      :class-card: sd-border-success gfo-compact

      ``return score``

      A single numeric value (int or float).
      Higher is better by default.


Basic Structure
---------------

.. code-block:: python

    def objective(params):
        # params is a dictionary: {"param1": value1, "param2": value2, ...}
        x = params["x"]
        y = params["y"]

        # Compute and return a score
        return -(x**2 + y**2)

The optimizer will call this function many times with different parameter
combinations, tracking which gives the best score.


Maximization vs. Minimization
-----------------------------

GFO **maximizes** the objective function by default. For minimization:

**Option 1: Negate the return value**

.. code-block:: python

    def objective(params):
        error = compute_error(params)
        return -error  # Negate for minimization

**Option 2: Use optimum="minimum"**

.. code-block:: python

    opt.search(objective, n_iter=100, optimum="minimum")


Return Values
-------------

The objective function should return a single numeric value (int or float).
Higher values are considered better.

.. code-block:: python

    # Good: returns a number
    def objective(params):
        return params["x"] ** 2

    # Also good: ML accuracy
    def objective(params):
        model = create_model(params)
        return cross_val_score(model, X, y).mean()


Handling Errors
---------------

If your objective function might fail for some parameter combinations,
handle exceptions gracefully:

.. code-block:: python

    def objective(params):
        try:
            result = expensive_computation(params)
            return result
        except Exception:
            return -float("inf")  # Return very bad score

GFO will continue searching despite failed evaluations.


ML Hyperparameter Example
-------------------------

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    def objective(params):
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42,  # Fixed for reproducibility
        )
        # Return cross-validation accuracy
        return cross_val_score(clf, X_train, y_train, cv=5).mean()


Expensive Functions
-------------------

For expensive objective functions:

1. **Enable memory**: ``opt.search(..., memory=True)`` caches evaluations
2. **Use SMBO algorithms**: Bayesian, TPE, or Forest learn from past evaluations
3. **Start with fewer iterations**: Validate the setup before long runs

.. code-block:: python

    # For expensive functions, use Bayesian optimization with caching
    from gradient_free_optimizers import BayesianOptimizer

    opt = BayesianOptimizer(search_space)
    opt.search(
        expensive_objective,
        n_iter=50,
        memory=True,
    )


Debugging Tips
--------------

Add logging to understand what's happening:

.. code-block:: python

    def objective(params):
        print(f"Evaluating: {params}")
        score = compute_score(params)
        print(f"  -> Score: {score}")
        return score

Or track evaluations manually:

.. code-block:: python

    evaluations = []

    def objective(params):
        score = compute_score(params)
        evaluations.append({"params": params, "score": score})
        return score
