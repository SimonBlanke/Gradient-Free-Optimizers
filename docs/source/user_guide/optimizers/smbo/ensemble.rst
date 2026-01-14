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

**Trade-offs:**

- More computational overhead (training multiple models)
- More complex to tune
- May average out good predictions with poor ones


Related Algorithms
------------------

- :doc:`bayesian` - Single GP model
- :doc:`forest` - Single tree ensemble
- :doc:`tpe` - Density-based approach
