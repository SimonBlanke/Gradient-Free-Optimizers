========================
Repulsing Hill Climbing
========================

Repulsing Hill Climbing extends standard Hill Climbing by **increasing the step
size** when the algorithm gets stuck. This allows it to "repulse" away from
local optima or flat regions.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/repulsing_hill_climbing_sphere_function_.gif
            :alt: Repulsing Hill Climbing on Sphere function

            **Convex function**: Converges normally when making progress.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/repulsing_hill_climbing_ackley_function_.gif
            :alt: Repulsing Hill Climbing on Ackley function

            **Multi-modal function**: Increases step size to escape
            local optima.


Algorithm
---------

At each iteration:

1. Generate neighbors within current ``epsilon`` distance
2. Evaluate neighbors
3. If improvement found:

   - Move to best neighbor
   - Reset epsilon to initial value

4. If no improvement:

   - Increase epsilon by ``repulsion_factor``
   - Larger steps help escape flat regions or local optima


The key insight: when stuck, take bigger steps to explore further away.
When making progress, take small steps for precision.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``repulsion_factor``
      - float
      - 5.0
      - Multiplier for epsilon when stuck
    * - ``epsilon``
      - float
      - 0.03
      - Initial step size
    * - ``distribution``
      - str
      - "normal"
      - Step distribution
    * - ``n_neighbours``
      - int
      - 3
      - Number of neighbors per iteration


The repulsion_factor Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``repulsion_factor = 1.0``: No increase (equivalent to standard Hill Climbing)
- ``repulsion_factor = 5.0``: Default, 5x increase when stuck
- ``repulsion_factor = 10.0``: Aggressive escape from local optima

The step size grows as: ``epsilon * (repulsion_factor ** n_stuck_iterations)``


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import RepulsingHillClimbingOptimizer

    def griewank(para):
        x, y = para["x"], para["y"]
        sum_sq = (x**2 + y**2) / 4000
        prod_cos = np.cos(x) * np.cos(y / np.sqrt(2))
        return -(sum_sq - prod_cos + 1)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = RepulsingHillClimbingOptimizer(
        search_space,
        repulsion_factor=3.0,  # Moderate repulsion
        epsilon=0.02,
    )

    opt.search(griewank, n_iter=500)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Functions with flat regions (plateaus)
- When standard Hill Climbing gets stuck repeatedly
- Problems where you don't want constant randomness

**Compared to other variants:**

- **vs. Stochastic HC**: Repulsing uses adaptive step size, not random acceptance
- **vs. Simulated Annealing**: Repulsing increases exploration when stuck,
  SA decreases it over time regardless of progress


Related Algorithms
------------------

- :doc:`hill_climbing` - Fixed step size
- :doc:`stochastic_hill_climbing` - Random acceptance of worse solutions
- :doc:`simulated_annealing` - Temperature-based decreasing acceptance
