===================
Spiral Optimization
===================

Spiral Optimization moves particles in spiral trajectories toward the global
best position. The spiral motion provides a balance between exploration
(outer spiral) and exploitation (inner spiral converging to center).


Algorithm
---------

Each particle follows a spiral path toward the global best:

1. Compute distance and angle to global best
2. Rotate position around global best by spiral angle
3. Move closer by decay factor
4. Update if new position is better

The spiral motion ensures particles explore the region around the best
solution while gradually converging.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``population``
      - int
      - 10
      - Number of particles
    * - ``decay_rate``
      - float
      - 0.99
      - How quickly spirals contract (closer to 1 = slower)


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import SpiralOptimization

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = SpiralOptimization(
        search_space,
        population=15,
        decay_rate=0.98,
    )

    opt.search(objective, n_iter=200)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Continuous optimization
- When you want balanced exploration around the best solution
- Problems where the optimum has a basin of attraction

**Compared to PSO:**

Spiral Optimization provides more structured exploration around the global
best, while PSO balances personal and global best attractions.


Related Algorithms
------------------

- :doc:`particle_swarm` - Velocity-based swarm movement
- :doc:`evolution_strategy` - Population of hill climbers
