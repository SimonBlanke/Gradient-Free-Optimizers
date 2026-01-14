================
Forest Optimizer
================

Forest Optimizer uses tree-based ensemble models (Random Forest or Extra Trees)
as surrogate models. It scales well to large search spaces and handles discrete
parameters naturally.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/forest_optimization_sphere_function_.gif
            :alt: Forest Optimizer on Sphere function

            **Convex function**: Tree ensemble guides the search.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/forest_optimization_ackley_function_.gif
            :alt: Forest Optimizer on Ackley function

            **Multi-modal function**: Handles complex landscapes.


Algorithm
---------

Similar to Bayesian Optimization, but using tree ensembles:

1. **Fit surrogate**: Train Random Forest/Extra Trees on observations
2. **Predict**: Get predictions from all trees in the ensemble
3. **Uncertainty**: Variance across tree predictions
4. **Acquisition**: Expected Improvement using mean and variance
5. **Select and evaluate**: Choose best acquisition, run objective


Why Trees?
----------

Tree-based models offer several advantages:

- **Scalability**: O(n log n) training vs. O(n^3) for GP
- **Discrete parameters**: Handle naturally (no encoding needed)
- **Non-stationarity**: Adapt to different regions of space
- **Robust**: Less sensitive to hyperparameter choices


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``tree_regressor``
      - str
      - "extra_tree"
      - Model type: "extra_tree", "random_forest", or "gradient_boost"
    * - ``tree_para``
      - dict
      - {"n_estimators": 100}
      - Parameters for the tree model
    * - ``xi``
      - float
      - 0.03
      - Exploration-exploitation trade-off


Tree Regressor Options
^^^^^^^^^^^^^^^^^^^^^^

- **extra_tree**: Extra Trees (randomized splits, faster)
- **random_forest**: Random Forest (classic, robust)
- **gradient_boost**: Gradient Boosting (sequential, powerful)

.. code-block:: python

    # Extra Trees (default, fast)
    opt = ForestOptimizer(search_space, tree_regressor="extra_tree")

    # Random Forest (more robust)
    opt = ForestOptimizer(
        search_space,
        tree_regressor="random_forest",
        tree_para={"n_estimators": 200}
    )


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ForestOptimizer
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)

    def objective(para):
        clf = SVC(
            C=para["C"],
            kernel=para["kernel"],
            gamma=para["gamma"],
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "C": np.logspace(-2, 2, 50),
        "kernel": np.array(["linear", "rbf", "poly"]),
        "gamma": np.logspace(-3, 0, 30),
    }

    opt = ForestOptimizer(
        search_space,
        tree_regressor="extra_tree",
        tree_para={"n_estimators": 100},
    )

    opt.search(objective, n_iter=50)
    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best params: {opt.best_para}")


When to Use
-----------

**Good for:**

- Large search spaces with many parameters
- Mixed discrete and continuous parameters
- When you need many evaluations (100+)
- Parallelizable (tree training is fast)

**Compared to other SMBO:**

- **vs. Bayesian**: Better scaling, worse uncertainty
- **vs. TPE**: Similar performance, different mechanism


Related Algorithms
------------------

- :doc:`bayesian` - GP-based, better uncertainty
- :doc:`tpe` - Density-based, similar use cases
- :doc:`ensemble` - Combines multiple models
