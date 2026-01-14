======================================
Tree-structured Parzen Estimators (TPE)
======================================

TPE is a sequential model-based optimization algorithm that models the objective
function using kernel density estimation. It's particularly effective for
hyperparameter optimization with categorical and conditional parameters.


Algorithm
---------

Unlike Bayesian Optimization which models f(x) directly, TPE models:

1. **l(x)**: Density of parameters where f(x) < threshold (good)
2. **g(x)**: Density of parameters where f(x) >= threshold (bad)

The acquisition function is:

.. code-block:: text

    EI(x) ~ l(x) / g(x)

Points are selected where the ratio is high (likely good, unlikely bad).


The Key Insight
---------------

TPE separates observations into "good" and "bad" groups based on a quantile
threshold (``gamma_tpe``). This approach:

- Works naturally with categorical parameters
- Handles conditional dependencies
- Scales better than GP to many observations


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``gamma_tpe``
      - float
      - 0.2
      - Fraction of observations considered "good"
    * - ``xi``
      - float
      - 0.03
      - Exploration-exploitation trade-off


The gamma_tpe Parameter
^^^^^^^^^^^^^^^^^^^^^^^

- **Lower gamma (0.1)**: Only top 10% are "good", more selective
- **Higher gamma (0.3)**: Top 30% are "good", more permissive

.. code-block:: python

    # Very selective (only best 10%)
    opt = TreeStructuredParzenEstimators(search_space, gamma_tpe=0.1)

    # More permissive (best 30%)
    opt = TreeStructuredParzenEstimators(search_space, gamma_tpe=0.3)


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import TreeStructuredParzenEstimators
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    def objective(para):
        clf = RandomForestClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            criterion=para["criterion"],
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=5).mean()

    search_space = {
        "n_estimators": np.arange(10, 200, 10),
        "max_depth": np.arange(2, 20),
        "criterion": np.array(["gini", "entropy"]),  # Categorical!
    }

    opt = TreeStructuredParzenEstimators(search_space)
    opt.search(objective, n_iter=50)

    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best params: {opt.best_para}")


When to Use
-----------

**Good for:**

- Mixed continuous, discrete, and categorical spaces
- Hyperparameter optimization with many categoricals
- When you have conditional parameters
- Larger evaluation budgets (scales better than GP)

**Compared to Bayesian Optimization:**

- TPE: Better for categoricals, faster for many observations
- BO: Better uncertainty quantification, often better for pure continuous


Related Algorithms
------------------

- :doc:`bayesian` - GP-based, better for continuous
- :doc:`forest` - Tree-based surrogate
