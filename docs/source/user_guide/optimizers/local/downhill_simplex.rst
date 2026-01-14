================
Downhill Simplex
================

The Downhill Simplex algorithm (also known as the Nelder-Mead method) uses a
geometric approach: it maintains a simplex (a shape with n+1 vertices in
n-dimensional space) and transforms it through reflection, expansion,
contraction, and shrinking operations to find the optimum.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/downhill_simplex_sphere_function_.gif
            :alt: Downhill Simplex on Sphere function

            **Convex function**: The simplex contracts toward the minimum.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/downhill_simplex_ackley_function_.gif
            :alt: Downhill Simplex on Ackley function

            **Multi-modal function**: May converge to local optimum
            depending on initial simplex placement.


Algorithm
---------

The algorithm maintains n+1 vertices for an n-dimensional problem. At each
iteration:

1. **Order** vertices by score (best to worst)
2. **Reflect** the worst vertex through the centroid of the remaining vertices
3. Depending on the reflection result:

   - If best so far: **Expand** further in that direction
   - If improvement: **Accept** the reflection
   - If still worst: **Contract** toward the centroid
   - If contraction fails: **Shrink** entire simplex toward best vertex

This geometric approach doesn't require gradient information but adapts to
the local curvature of the objective function.


Simplex Operations
------------------

.. code-block:: text

    Given vertices: [best, ..., worst] and centroid C of [best, ..., second_worst]

    Reflection:   R = C + alpha * (C - worst)      # Try opposite side
    Expansion:    E = C + gamma * (R - C)          # Go further if R is good
    Contraction:  K = C + beta * (worst - C)       # Pull back toward center
    Shrink:       x_i = best + 0.5 * (x_i - best)  # Contract entire simplex


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``alpha``
      - float
      - 1.0
      - Reflection coefficient (typically 1.0)
    * - ``gamma``
      - float
      - 2.0
      - Expansion coefficient (typically 2.0)
    * - ``beta``
      - float
      - 0.5
      - Contraction coefficient (typically 0.5)


Standard Coefficients
^^^^^^^^^^^^^^^^^^^^^

The default values (alpha=1.0, gamma=2.0, beta=0.5) are the original Nelder-Mead
coefficients and work well for most problems. Adjust only if you have specific
needs:

- **Larger alpha/gamma**: More aggressive exploration
- **Smaller beta**: More conservative contraction
- **0.5 shrink factor**: Standard (not configurable in GFO)


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import DownhillSimplexOptimizer

    def rosenbrock(para):
        x, y = para["x"], para["y"]
        return -((1 - x)**2 + 100 * (y - x**2)**2)

    search_space = {
        "x": np.linspace(-5, 5, 100),
        "y": np.linspace(-5, 5, 100),
    }

    opt = DownhillSimplexOptimizer(
        search_space,
        alpha=1.0,   # Standard reflection
        gamma=2.0,   # Standard expansion
        beta=0.5,    # Standard contraction
    )

    opt.search(rosenbrock, n_iter=500)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Low-dimensional problems (< 10 dimensions)
- Smooth objective functions
- When you want a deterministic, geometry-based approach
- Problems where function evaluations are cheap

**Not ideal for:**

- High-dimensional spaces (simplex has n+1 vertices)
- Noisy objective functions
- Functions with many local optima
- Discrete or categorical parameters


Comparison with Other Local Methods
-----------------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - vs. Hill Climbing
      - Simplex uses n+1 points vs. neighborhood sampling. Better for smooth
        functions, worse for discrete spaces.
    * - vs. Powell's Method
      - Simplex optimizes all dimensions jointly, Powell optimizes sequentially.
        Simplex is better for coupled parameters.


Initialization Note
-------------------

The Downhill Simplex requires n+1 initial points. In GFO, these are generated
from the initialization strategy (grid, random, vertices, or warm_start).
Ensure you have at least n+1 initial positions:

.. code-block:: python

    n_dims = len(search_space)

    opt = DownhillSimplexOptimizer(
        search_space,
        initialize={"random": n_dims + 1}  # Ensure enough initial points
    )


Related Algorithms
------------------

- :doc:`hill_climbing` - Simpler neighborhood-based local search
- :doc:`../global/powells_method` - Sequential 1D optimization
- :doc:`../global/pattern_search` - Structured geometric exploration
