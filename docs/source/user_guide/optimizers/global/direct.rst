================
DIRECT Algorithm
================

DIRECT (DIviding RECTangles) is a deterministic global optimization algorithm
that recursively divides the search space into smaller regions, focusing on
the most promising areas while ensuring global coverage.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/direct_algorithm_sphere_function_.gif
            :alt: DIRECT on Sphere function

            **Convex function**: Hierarchical division focuses quickly.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/direct_algorithm_ackley_function_.gif
            :alt: DIRECT on Ackley function

            **Multi-modal function**: Balanced exploration and exploitation.


Algorithm
---------

DIRECT works by:

1. Evaluate the center of the search space
2. Divide the space into smaller hyperrectangles
3. For each rectangle, compute a potential based on:

   - The function value at the center
   - The size of the rectangle (larger = more unexplored)

4. Select **potentially optimal** rectangles (Pareto-optimal in value vs. size)
5. Divide selected rectangles and repeat

This approach balances:

- **Exploitation**: Focus on regions with good values
- **Exploration**: Continue investigating large, unexplored regions


Key Properties
--------------

- **Deterministic**: Same results for same input
- **Global convergence**: Guaranteed to find global optimum as iterations increase
- **No hyperparameters**: Self-adapting division strategy
- **Lipschitz-free**: Doesn't require knowledge of Lipschitz constant


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
      - Sampling strategy for candidates
    * - ``max_sample_size``
      - int
      - 10000000
      - Maximum sample size


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import DirectAlgorithm

    def schwefel(para):
        x, y = para["x"], para["y"]
        return (x * np.sin(np.sqrt(abs(x))) +
                y * np.sin(np.sqrt(abs(y))))

    search_space = {
        "x": np.linspace(-500, 500, 1000),
        "y": np.linspace(-500, 500, 1000),
    }

    opt = DirectAlgorithm(search_space)
    opt.search(schwefel, n_iter=100)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Global optimization with convergence guarantees
- Unknown landscapes where you need both exploration and exploitation
- Problems where you want deterministic, reproducible results

**Not ideal for:**

- Very high-dimensional spaces (rectangle count grows exponentially)
- Problems with cheap evaluations where simpler methods suffice


Comparison with Other Global Methods
------------------------------------

.. list-table::
    :header-rows: 1

    * - Algorithm
      - Approach
      - Guarantees
    * - DIRECT
      - Hierarchical division
      - Global convergence
    * - Lipschitz
      - Bound-based pruning
      - Requires smoothness
    * - Random Search
      - Pure random
      - Probabilistic
    * - Bayesian
      - Surrogate model
      - None (model-based)


Related Algorithms
------------------

- :doc:`lipschitz` - Related bounding approach
- :doc:`grid_search` - Non-adaptive systematic coverage
- :doc:`../smbo/bayesian` - Model-based global optimization
