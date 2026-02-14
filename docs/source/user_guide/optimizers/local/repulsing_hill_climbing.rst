========================
Repulsing Hill Climbing
========================

Repulsing Hill Climbing extends Hill Climbing with an adaptive step size that
grows when the algorithm stagnates. While making progress, it behaves identically
to standard Hill Climbing, using a fixed ``epsilon`` step size. When no improving
neighbor is found, the algorithm multiplies the current step size by a
``repulsion_factor`` on each subsequent iteration. This growth
continues until an improving position is reached, at which point the step size
resets to its initial value. The result is an automatic alternation between
fine-grained local search and large exploratory jumps.


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


The escape mechanism here is deterministic and reactive, which distinguishes it
from both Stochastic Hill Climbing (probabilistic acceptance of worse solutions)
and Simulated Annealing (time-dependent acceptance schedule). Repulsing Hill
Climbing does not accept worse solutions at all; instead, it increases its search
radius until it finds a better position elsewhere. This makes it well suited for
landscapes with flat plateaus or wide basins of attraction where small
perturbations cannot escape the current region. Choose it when the search space
topology is unknown and you want the optimizer to self-adapt its exploration
radius without requiring manual parameter scheduling.


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


.. code-block:: text

    if improvement found:
        epsilon = initial_epsilon            # Reset to precise steps
    else:
        epsilon = epsilon * repulsion_factor  # Exponential growth

.. note::

    The step size grows **exponentially** when stuck
    (``epsilon * repulsion_factor^n``), which means the algorithm can very
    quickly escape even wide plateaus. But the instant an improvement is found,
    epsilon resets to its initial value for precise local search. This
    all-or-nothing behavior makes it aggressive at escaping but precise
    when making progress.


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


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import RepulsingHillClimbingOptimizer

    def plateau_function(para):
        """Function with large flat regions and a narrow optimum."""
        x, y, z = para["x"], para["y"], para["z"]
        return -(np.round(x**2 + y**2 + z**2, 1))

    search_space = {
        "x": np.linspace(-10, 10, 200),
        "y": np.linspace(-10, 10, 200),
        "z": np.linspace(-10, 10, 200),
    }

    opt = RepulsingHillClimbingOptimizer(
        search_space,
        repulsion_factor=5.0,
        epsilon=0.02,
        n_neighbours=5,
    )

    opt.search(plateau_function, n_iter=1000)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: Automatically adaptive. The algorithm is
  exploitative by default and only becomes exploratory when stuck.
- **Computational overhead**: Same as Hill Climbing (minimal).
- **Parameter sensitivity**: ``repulsion_factor`` determines how aggressively the
  algorithm escapes. Very high values can cause it to jump far from good regions.


Related Algorithms
------------------

- :doc:`hill_climbing` - Fixed step size
- :doc:`stochastic_hill_climbing` - Random acceptance of worse solutions
- :doc:`simulated_annealing` - Temperature-based decreasing acceptance
