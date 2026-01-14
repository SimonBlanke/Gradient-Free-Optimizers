=====================
Simple Optimization
=====================

This example demonstrates the basic usage of Gradient-Free-Optimizers on
a simple 2D function.


Minimizing a Quadratic Function
-------------------------------

Let's find the minimum of the sphere function: f(x, y) = x^2 + y^2

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    # Define the objective function
    # GFO maximizes by default, so we negate for minimization
    def sphere(para):
        x = para["x"]
        y = para["y"]
        return -(x**2 + y**2)

    # Define the search space as NumPy arrays
    search_space = {
        "x": np.linspace(-10, 10, 100),  # 100 values from -10 to 10
        "y": np.linspace(-10, 10, 100),
    }

    # Create optimizer and run search
    opt = HillClimbingOptimizer(search_space)
    opt.search(sphere, n_iter=500)

    # Results
    print(f"Best parameters: {opt.best_para}")
    print(f"Best score: {opt.best_score}")
    print(f"Evaluations: {len(opt.search_data)}")

**Expected output:**

.. code-block:: text

    Best parameters: {'x': 0.0, 'y': 0.0}
    Best score: -0.0
    Evaluations: 500


Multi-Modal Function (Rastrigin)
--------------------------------

The Rastrigin function has many local optima, making it harder to optimize:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import SimulatedAnnealingOptimizer

    def rastrigin(para):
        x = para["x"]
        y = para["y"]
        A = 10
        return -(A * 2 + (x**2 - A * np.cos(2 * np.pi * x))
                      + (y**2 - A * np.cos(2 * np.pi * y)))

    search_space = {
        "x": np.linspace(-5.12, 5.12, 200),
        "y": np.linspace(-5.12, 5.12, 200),
    }

    # Simulated Annealing can escape local optima
    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=0.98,
        start_temp=1.0,
    )
    opt.search(rastrigin, n_iter=2000)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")
    # Global optimum is at (0, 0) with score 0


Higher Dimensions
-----------------

GFO handles higher-dimensional spaces:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ParticleSwarmOptimizer

    def sphere_nd(para):
        return -sum(para[f"x{i}"]**2 for i in range(5))

    # 5-dimensional search space
    search_space = {
        f"x{i}": np.linspace(-10, 10, 50)
        for i in range(5)
    }

    opt = ParticleSwarmOptimizer(search_space, population=20)
    opt.search(sphere_nd, n_iter=500)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Accessing All Results
---------------------

The ``search_data`` attribute contains all evaluations:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gradient_free_optimizers import RandomSearchOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective, n_iter=100)

    # Get all evaluations as DataFrame
    df = opt.search_data
    print(df.head())
    print(f"\nBest 5 results:")
    print(df.nlargest(5, "score"))
