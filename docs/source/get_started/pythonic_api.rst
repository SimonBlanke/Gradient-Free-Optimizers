============
Pythonic API
============

Gradient-Free-Optimizers uses standard Python data structures throughout its
API. There are no special configuration objects, no domain-specific languages,
and no wrapper classes to learn. If you know Python dicts, callables, and
pandas DataFrames, you already know the API.


Three Key Interfaces
--------------------

The entire workflow uses three concepts, each mapped to a familiar Python type:

**1. Search space = Python dict with NumPy arrays**

.. code-block:: python

    import numpy as np

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.arange(1, 50),
        "method": np.array(["adam", "sgd", "rmsprop"]),
    }

**2. Objective = any Python callable that takes a dict and returns a float**

.. code-block:: python

    def objective(para):
        x = para["x"]
        y = para["y"]
        return -(x ** 2 + y ** 2)

**3. Results = pandas DataFrame via** ``opt.search_data``

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=100)

    # pandas DataFrame with all evaluated positions and scores
    print(opt.search_data)


Complete Example
----------------

Here is a full optimization from definition to results:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    # Search space: plain dict
    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # Objective: plain function
    def objective(para):
        return -(para["x"] ** 2 + para["y"] ** 2)

    # Run
    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=1000)

    # Access results
    print(f"Best parameters: {opt.best_para}")    # dict
    print(f"Best score:      {opt.best_score}")    # float
    print(f"All evaluations: {len(opt.search_data)} rows")  # DataFrame


Algorithm Swapping
------------------

Because every optimizer shares the same constructor and ``search()`` signature,
switching algorithms requires changing only the class name. Nothing else in your
code needs to change:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import (
        HillClimbingOptimizer,
        BayesianOptimizer,
        ParticleSwarmOptimizer,
    )

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    def objective(para):
        return -(para["x"] ** 2 + para["y"] ** 2)

    # Same code, different algorithm
    for Optimizer in [HillClimbingOptimizer, BayesianOptimizer, ParticleSwarmOptimizer]:
        opt = Optimizer(search_space)
        opt.search(objective, n_iter=100, verbosity=False)
        print(f"{Optimizer.__name__:>30s}: best = {opt.best_score:.4f}")

Population-based optimizers accept an additional ``population`` parameter, and
surrogate-model-based optimizers accept model configuration parameters, but the
core interface remains identical.


Accessing Results
-----------------

After a search completes, results are available through three properties:

``opt.best_para``
    A dictionary with the best parameter combination found.

``opt.best_score``
    A float with the best objective value achieved.

``opt.search_data``
    A pandas DataFrame containing every evaluated position and its score.
    Column names match the search space keys, plus a ``score`` column.

.. code-block:: python

    # Best result
    print(opt.best_para)
    # {'x': 0.10101010101010033, 'y': 0.10101010101010033}

    print(opt.best_score)
    # -0.020408163265306273

    # Full history as DataFrame
    df = opt.search_data
    print(df.columns.tolist())
    # ['x', 'y', 'score']

    # Standard pandas operations work
    top_5 = df.nlargest(5, "score")
    print(top_5)
