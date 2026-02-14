====================
Bayesian Optimization
====================

Bayesian Optimization uses a Gaussian Process (GP) as a surrogate model to
predict the objective function and guide the search toward promising regions.
It's particularly effective for expensive objective functions.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/bayesian_optimization_sphere_function_.gif
            :alt: Bayesian Optimization on Sphere function

            **Convex function**: Efficiently narrows down to the optimum.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/bayesian_optimization_ackley_function_.gif
            :alt: Bayesian Optimization on Ackley function

            **Multi-modal function**: Balances exploration and exploitation.


Algorithm
---------

At each iteration:

1. **Fit surrogate model**: Train GP on all (position, score) observations
2. **Predict**: For candidate positions, predict mean and uncertainty
3. **Acquisition function**: Compute Expected Improvement (EI)

   .. code-block:: text

       EI(x) = E[max(0, f(x) - f(x_best))]

   EI is high when: predicted value is good OR uncertainty is high

4. **Select**: Choose position with highest acquisition value
5. **Evaluate**: Run actual objective function
6. **Update**: Add new observation, repeat

.. note::

    **Key Insight:** Bayesian Optimization is fundamentally about making
    decisions under uncertainty. The GP surrogate provides not just a prediction
    but a full probability distribution at every point. The Expected Improvement
    acquisition function then naturally balances exploration (high uncertainty)
    and exploitation (high predicted value) in a single, principled formula.
    This is why BO can find good solutions in remarkably few evaluations
    compared to methods that lack an uncertainty model.

.. figure:: /_static/diagrams/bayesian_optimization_flowchart.svg
    :alt: Bayesian Optimization algorithm flowchart
    :align: center

    The BO loop: fit GP, compute Expected Improvement, select the point
    with highest acquisition value, evaluate, and update the dataset.


The Gaussian Process
--------------------

A GP provides:

- **Mean prediction**: Expected function value at each point
- **Uncertainty**: How confident we are (from covariance)

This uncertainty is crucial: it allows the algorithm to explore
uncertain regions that might contain the optimum.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``xi``
      - float
      - 0.03
      - Exploration-exploitation trade-off
    * - ``gpr``
      - object
      - gaussian_process["gp_nonlinear"]
      - Gaussian Process regressor configuration


The xi Parameter
^^^^^^^^^^^^^^^^

``xi`` controls the exploration-exploitation balance:

- **Lower xi (0.01)**: Focus on regions with high predicted scores
- **Higher xi (0.1)**: Explore uncertain regions more

.. code-block:: python

    # Exploitation-focused
    opt = BayesianOptimizer(search_space, xi=0.01)

    # Exploration-focused
    opt = BayesianOptimizer(search_space, xi=0.1)


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)

    def objective(para):
        clf = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            learning_rate=para["learning_rate"],
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "n_estimators": np.arange(50, 200, 10),
        "max_depth": np.arange(2, 10),
        "learning_rate": np.linspace(0.01, 0.3, 20),
    }

    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=30)

    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best params: {opt.best_para}")


When to Use
-----------

**Good for:**

- Expensive objective functions (ML training, simulations)
- Low-dimensional continuous spaces (< 20 dimensions)
- When you have limited evaluation budget
- Problems where each evaluation takes seconds to hours

**Not ideal for:**

- Very cheap functions (GP overhead dominates)
- Very high dimensions (GP scales poorly)
- Purely discrete/categorical spaces (consider TPE)


Computational Cost
------------------

GP training is O(n^3) where n is the number of observations:

- 10 observations: negligible
- 100 observations: noticeable
- 1000 observations: significant

For many evaluations, consider :doc:`forest` or :doc:`tpe`.


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)

    def objective(para):
        clf = KNeighborsClassifier(
            n_neighbors=para["n_neighbors"],
            weights=para["weights"],
            p=para["p"],
            leaf_size=para["leaf_size"],
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "n_neighbors": np.arange(1, 30),
        "weights": np.array(["uniform", "distance"]),
        "p": np.array([1, 2]),
        "leaf_size": np.arange(10, 60, 5),
    }

    opt = BayesianOptimizer(search_space, xi=0.05)
    opt.search(objective, n_iter=40)

    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best params: {opt.best_para}")


Trade-offs
----------

- **Exploration vs. exploitation**: Controlled by ``xi``. The GP's uncertainty
  naturally decays in well-sampled regions, so exploration shifts automatically
  toward unexplored areas.
- **Computational overhead**: GP training is O(n^3) in observations. This makes
  BO best suited for problems where the objective function is far more expensive
  than the surrogate model fitting.
- **Parameter sensitivity**: The default ``xi=0.03`` works well for most problems.
  The choice of GP kernel (via ``gpr``) can matter more than ``xi`` for complex
  landscapes.


Related Algorithms
------------------

- :doc:`tpe` - Density-based, handles categoricals well
- :doc:`forest` - Tree-based, scales better
- :doc:`ensemble` - Combines multiple models
