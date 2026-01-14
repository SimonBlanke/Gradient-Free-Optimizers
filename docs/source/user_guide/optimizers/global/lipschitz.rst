===================
Lipschitz Optimizer
===================

The Lipschitz Optimizer uses **Lipschitz continuity bounds** to prune the search
space. By assuming the objective function has a bounded rate of change, it can
eliminate regions that cannot contain the optimum.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/lipschitz_optimizer_sphere_function_.gif
            :alt: Lipschitz Optimizer on Sphere function

            **Convex function**: Uses bounds to focus search.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/lipschitz_optimizer_ackley_function_.gif
            :alt: Lipschitz Optimizer on Ackley function

            **Multi-modal function**: Bounds help prune unpromising regions.


Algorithm
---------

The Lipschitz condition states that for any two points x and y:

.. code-block:: text

    |f(x) - f(y)| <= L * ||x - y||

where L is the Lipschitz constant. This provides upper and lower bounds on
function values at unevaluated points.

The algorithm:

1. Sample candidate points from the search space
2. For each candidate, compute bounds based on nearby evaluated points
3. Select the point with the best potential (highest upper bound)
4. Evaluate and add to the set of observations


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``sampling``
      - dict
      - {"random": 1000000}
      - How many candidate samples to consider
    * - ``max_sample_size``
      - int
      - 10000000
      - Maximum sample size for efficiency


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import LipschitzOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = LipschitzOptimizer(
        search_space,
        sampling={"random": 100000},
    )

    opt.search(objective, n_iter=50)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Functions with known or assumed smoothness
- When you want theoretical guarantees
- Problems where pruning can significantly reduce search space

**Not ideal for:**

- Discontinuous or very noisy functions
- Functions with unknown Lipschitz constant
- Very cheap function evaluations (overhead may dominate)


Related Algorithms
------------------

- :doc:`direct` - Related divide-and-conquer approach
- :doc:`../smbo/bayesian` - Model-based bounding with uncertainty
