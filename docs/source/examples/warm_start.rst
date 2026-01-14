======================
Memory and Warm Starts
======================

GFO supports caching evaluations and continuing from previous results.


Memory Caching
--------------

Enable caching to avoid re-evaluating the same positions:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    evaluation_count = 0

    def expensive_objective(para):
        global evaluation_count
        evaluation_count += 1
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 20),  # Coarse grid
        "y": np.linspace(-10, 10, 20),
    }

    # With memory=True, duplicate positions aren't re-evaluated
    opt = HillClimbingOptimizer(search_space)
    opt.search(expensive_objective, n_iter=500, memory=True)

    print(f"Iterations: 500")
    print(f"Actual evaluations: {evaluation_count}")
    print(f"Cache hits: {500 - evaluation_count}")


Warm Start from Previous Run
----------------------------

Continue optimization from a previous run's results:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # First optimization run
    opt1 = BayesianOptimizer(search_space, random_state=42)
    opt1.search(objective, n_iter=25)
    print(f"Run 1: best = {opt1.best_score:.4f}")

    # Get the search data from first run
    previous_data = opt1.search_data

    # Second run, starting from previous results
    opt2 = BayesianOptimizer(search_space, random_state=42)
    opt2.search(
        objective,
        n_iter=25,
        memory_warm_start=previous_data,  # Pass previous results
    )
    print(f"Run 2: best = {opt2.best_score:.4f}")

    # The second run benefits from knowledge of the first run


Warm Start with Specific Points
-------------------------------

Start from known good configurations:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # Initialize with specific starting points
    opt = HillClimbingOptimizer(
        search_space,
        initialize={
            "warm_start": [
                {"x": 1.0, "y": 1.0},   # Known good region
                {"x": -0.5, "y": 0.5},  # Another starting point
            ],
            "random": 2,  # Plus some random exploration
        }
    )

    opt.search(objective, n_iter=100)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


Resuming After Interruption
---------------------------

Save and restore optimization state:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gradient_free_optimizers import BayesianOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # First session
    opt1 = BayesianOptimizer(search_space)
    opt1.search(objective, n_iter=20)

    # Save results
    opt1.search_data.to_csv("optimization_checkpoint.csv", index=False)
    print(f"Session 1 complete: {opt1.best_score:.4f}")

    # Later session - resume from saved results
    saved_data = pd.read_csv("optimization_checkpoint.csv")

    opt2 = BayesianOptimizer(search_space)
    opt2.search(
        objective,
        n_iter=30,
        memory_warm_start=saved_data,
    )
    print(f"Session 2 complete: {opt2.best_score:.4f}")


Combining Multiple Search Histories
-----------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gradient_free_optimizers import (
        RandomSearchOptimizer,
        BayesianOptimizer,
    )

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # Run random search for exploration
    opt_random = RandomSearchOptimizer(search_space)
    opt_random.search(objective, n_iter=50)

    # Continue with Bayesian optimization
    opt_bayes = BayesianOptimizer(search_space)
    opt_bayes.search(
        objective,
        n_iter=30,
        memory_warm_start=opt_random.search_data,
    )

    print(f"Random search best: {opt_random.best_score:.4f}")
    print(f"Bayesian (with warm start) best: {opt_bayes.best_score:.4f}")
