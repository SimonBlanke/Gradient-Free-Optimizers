===========
Grid Search
===========

Grid Search systematically evaluates positions on a regular grid across the
search space. It guarantees complete coverage but suffers from the curse of
dimensionality.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/grid_search_sphere_function_.gif
            :alt: Grid Search on Sphere function

            **Convex function**: Systematic traversal of the grid.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/grid_search_ackley_function_.gif
            :alt: Grid Search on Ackley function

            **Multi-modal function**: Guaranteed to find the global
            optimum if it's on the grid.


Algorithm
---------

Grid Search traverses positions in a structured pattern:

1. Start at a corner of the search space
2. Move to adjacent grid positions following the specified direction
3. Evaluate each position
4. Track the best found

.. code-block:: text

    for each position on grid (step_size spacing):
        score = objective(position)
        if score > best_score:
            best_score, best_pos = score, position

GFO supports two traversal patterns:

- **Diagonal**: Moves diagonally across the grid (default)
- **Orthogonal**: Moves along axes, one dimension at a time

.. note::

    **Key Insight:** Grid Search is the only algorithm that guarantees finding
    the global optimum (if it lies on a grid point) given enough iterations.
    However, it projects poorly to important dimensions: in a 5D space where
    only 2 dimensions matter, grid search wastes most evaluations on
    irrelevant combinations.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``step_size``
      - int
      - 1
      - Grid spacing (1 = every point, 2 = every other point, etc.)
    * - ``direction``
      - str
      - "diagonal"
      - Traversal pattern: "diagonal" or "orthogonal"


Step Size
^^^^^^^^^

The ``step_size`` controls how many grid points to skip:

.. code-block:: python

    # Dense grid - evaluate every point
    opt = GridSearchOptimizer(search_space, step_size=1)

    # Sparse grid - skip every other point
    opt = GridSearchOptimizer(search_space, step_size=2)

.. warning::

    Total evaluations = product of (dimension_size / step_size) for all dimensions.
    For a 3D space with 100 points each and step_size=1, that's 1,000,000 evaluations!


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import GridSearchOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    # Small search space for grid search
    search_space = {
        "x": np.linspace(-5, 5, 20),  # 20 points
        "y": np.linspace(-5, 5, 20),  # 20 points
    }
    # Total: 400 evaluations needed for complete coverage

    opt = GridSearchOptimizer(search_space, step_size=1)
    opt.search(objective, n_iter=400)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Small, low-dimensional search spaces
- When you need guaranteed coverage
- Discrete parameter spaces with few options
- When reproducibility is critical

**Not ideal for:**

- High-dimensional spaces (exponential growth)
- Large continuous spaces
- Expensive function evaluations


The Curse of Dimensionality
---------------------------

Grid Search doesn't scale well with dimensions:

.. list-table::
    :header-rows: 1

    * - Dimensions
      - Points per dim
      - Total evaluations
    * - 2
      - 10
      - 100
    * - 3
      - 10
      - 1,000
    * - 5
      - 10
      - 100,000
    * - 10
      - 10
      - 10,000,000,000

For high-dimensional problems, consider Random Search or SMBO algorithms.


Comparison with Random Search
-----------------------------

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * - Aspect
      - Grid Search
      - Random Search
    * - Coverage
      - Complete (if enough iterations)
      - Probabilistic
    * - High dimensions
      - Curse of dimensionality
      - Scales linearly
    * - Parameter importance
      - Same resolution everywhere
      - Adapts through projection
    * - Early stopping
      - May miss optimum
      - Probabilistically fair


Sparse Grid Example
-------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import GridSearchOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2 + para["z"]**2)

    # Coarse grid to manage 3D space
    search_space = {
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-5, 5, 10),
        "z": np.linspace(-5, 5, 10),
    }

    opt = GridSearchOptimizer(search_space, step_size=2)
    opt.search(objective, n_iter=200)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: Pure exploration with complete coverage
  but no adaptation to promising regions.
- **Computational overhead**: Effectively zero per step, but total cost
  grows exponentially with dimensions.
- **Parameter sensitivity**: ``step_size`` trades resolution for speed.
  Larger steps risk missing the optimum between grid points.


Related Algorithms
------------------

- :doc:`random_search` - Probabilistic coverage
- :doc:`pattern_search` - Structured but adaptive exploration
- :doc:`direct` - Adaptive grid refinement
