==================
Ensemble Optimizer
==================

The Ensemble Optimizer combines multiple surrogate models to create more robust
predictions. By averaging predictions from different model types, it reduces
the risk of any single model's weaknesses affecting the search.


Algorithm
---------

At each iteration:

1. **Fit multiple surrogates**: Train each model in the ensemble
2. **Combine predictions**: Average (or weighted average) predictions
3. **Compute uncertainty**: From ensemble disagreement
4. **Acquisition**: Expected Improvement using combined predictions
5. **Select and evaluate**: Standard SMBO loop

.. code-block:: text

    predictions = [model.predict(candidates) for model in estimators]
    mean_pred = average(predictions)
    uncertainty = variance(predictions)
    EI = expected_improvement(mean_pred, uncertainty, best_score, xi)
    next_point = argmax(EI)

.. note::

    **Key Insight:** The ensemble's uncertainty estimate comes from model
    disagreement rather than a single model's internal uncertainty. When a GP
    and a Random Forest disagree about a region, that disagreement signals
    genuine uncertainty that neither model alone would capture. This
    "epistemic diversity" makes the ensemble more robust to model
    misspecification than any single surrogate.


Why Ensemble?
-------------

Different models have different strengths:

- **GP**: Good uncertainty, struggles with non-stationarity
- **Random Forest**: Robust, handles discrete well
- **Gradient Boosting**: Powerful predictions, less uncertainty

Combining them provides:

- More robust predictions
- Better uncertainty estimates
- Reduced model-specific bias


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``estimators``
      - list
      - [GradientBoostingRegressor, GaussianProcessRegressor]
      - List of sklearn estimator classes
    * - ``xi``
      - float
      - 0.01
      - Exploration-exploitation trade-off


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import EnsembleOptimizer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor

    def objective(para):
        x, y = para["x"], para["y"]
        return -(x**2 + y**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = EnsembleOptimizer(
        search_space,
        estimators=[
            GradientBoostingRegressor,
            RandomForestRegressor,
            GaussianProcessRegressor,
        ],
        xi=0.02,
    )

    opt.search(objective, n_iter=50)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- When you're unsure which surrogate model is best
- Complex landscapes that might fool single models
- When robustness is more important than speed

**Not ideal for:**

- Tight computational budgets (training multiple models)
- Problems where a single model type is clearly best
- Simple, low-dimensional continuous functions (GP alone is sufficient)


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import EnsembleOptimizer
    from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor

    def schwefel_4d(para):
        vals = [para[f"x{i}"] for i in range(4)]
        return -sum(418.9829 - v * np.sin(np.sqrt(abs(v))) for v in vals)

    search_space = {
        f"x{i}": np.linspace(-500, 500, 200)
        for i in range(4)
    }

    opt = EnsembleOptimizer(
        search_space,
        estimators=[
            GradientBoostingRegressor,
            ExtraTreesRegressor,
            GaussianProcessRegressor,
        ],
        xi=0.03,
    )

    opt.search(schwefel_4d, n_iter=80)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: The ensemble naturally explores through model
  disagreement. ``xi`` provides additional control. The diversity of estimator
  types matters more than the number of estimators.
- **Computational overhead**: Linear in the number of estimators. Each estimator
  is trained independently, so the slowest model dominates wall-clock time.
- **Parameter sensitivity**: The choice of estimator types is the most important
  decision. Including models with different inductive biases (e.g., GP + tree-based)
  is more valuable than multiple similar models.


Related Algorithms
------------------

- :doc:`bayesian` - Single GP model
- :doc:`forest` - Single tree ensemble
- :doc:`tpe` - Density-based approach
