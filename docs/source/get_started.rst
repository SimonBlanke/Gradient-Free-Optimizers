:html_theme.sidebar_secondary.remove:

===========
Get Started
===========

This guide will help you install Gradient-Free-Optimizers and run your first optimization
in under 5 minutes.


Installation
------------

Install from PyPI using pip:

.. code-block:: bash

    pip install gradient-free-optimizers

**Requirements:**

- Python 3.10+
- NumPy, SciPy, pandas, tqdm

**Optional dependencies:**

For surrogate model-based optimizers with sklearn estimators:

.. code-block:: bash

    pip install gradient-free-optimizers[sklearn]


Your First Optimization
-----------------------

Let's optimize a simple 2D function to find its minimum:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    # 1. Define your objective function
    # The optimizer will try to MAXIMIZE this score
    def objective(para):
        x = para["x"]
        y = para["y"]
        # Negative because we want to minimize x^2 + y^2
        return -(x**2 + y**2)

    # 2. Define the search space
    # Each dimension is a NumPy array of possible values
    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # 3. Create an optimizer and run the search
    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=1000)

    # 4. Get the results
    print(f"Best parameters: {opt.best_para}")
    print(f"Best score: {opt.best_score}")

**Expected output:**

.. code-block:: text

    Best parameters: {'x': 0.0, 'y': 0.0}
    Best score: -0.0

.. tip::

    By default, GFO **maximizes** the objective function. To minimize, either:

    - Return the negative of your function (as shown above)
    - Use ``optimum="minimum"`` in ``search()``


Understanding Search Spaces
---------------------------

Search spaces in GFO are defined as dictionaries where:

- **Keys** are parameter names
- **Values** are NumPy arrays of possible values

.. code-block:: python

    search_space = {
        # Continuous: 100 values from 0.001 to 1.0
        "learning_rate": np.linspace(0.001, 1.0, 100),

        # Discrete integers: 10, 20, 30, ..., 200
        "n_estimators": np.arange(10, 210, 10),

        # Categorical: array of strings
        "optimizer": np.array(["adam", "sgd", "rmsprop"]),

        # Boolean: array with True/False
        "use_bias": np.array([True, False]),
    }

The optimizer samples from these arrays, so the **granularity** of your array
determines how precisely you can tune each parameter.


Choosing an Optimizer
---------------------

Here's a quick guide to help you choose:

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Scenario
      - Recommended
      - Why
    * - Fast baseline
      - ``RandomSearchOptimizer``
      - No overhead, establishes baseline performance
    * - Smooth functions
      - ``HillClimbingOptimizer``
      - Fast, effective for convex problems
    * - Many local optima
      - ``SimulatedAnnealingOptimizer``
      - Can escape local optima via temperature
    * - Expensive evaluations
      - ``BayesianOptimizer``
      - Learns from past evaluations
    * - Large populations
      - ``ParticleSwarmOptimizer``
      - Parallel exploration of search space
    * - Systematic coverage
      - ``GridSearchOptimizer``
      - Guarantees coverage of search space


Real-World Example: Hyperparameter Tuning
-----------------------------------------

Here's a practical example tuning a Random Forest classifier:

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    from gradient_free_optimizers import BayesianOptimizer

    # Load data
    X, y = load_iris(return_X_y=True)

    # Define objective: return cross-validation accuracy
    def objective(para):
        clf = RandomForestClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=5).mean()

    # Define search space
    search_space = {
        "n_estimators": np.arange(10, 200, 10),
        "max_depth": np.arange(2, 20),
        "min_samples_split": np.arange(2, 20),
    }

    # Run Bayesian optimization
    opt = BayesianOptimizer(search_space)
    opt.search(
        objective,
        n_iter=50,
        verbosity=["progress_bar", "print_results"],
    )

    # Results
    print(f"\nBest accuracy: {opt.best_score:.4f}")
    print(f"Best parameters: {opt.best_para}")

    # Access all evaluations as a DataFrame
    print(f"\nAll evaluations:")
    print(opt.search_data.head())


Using the Ask-Tell Interface
----------------------------

For more control over the optimization loop, use the ask-tell interface:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ParticleSwarmOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # Create optimizer and set up the search
    opt = ParticleSwarmOptimizer(search_space, population=10)
    opt.setup_search(objective, n_iter=100)

    # Manual optimization loop
    for iteration in range(100):
        # Get next parameters to evaluate
        params = opt.ask()

        # Evaluate (you could do this anywhere, even distributed)
        score = objective(params)

        # Report result back to optimizer
        opt.tell(params, score)

        # You have full control here: logging, early stopping, etc.
        if iteration % 25 == 0:
            print(f"Iteration {iteration}: best = {opt.best_score:.4f}")

    print(f"\nFinal best: {opt.best_score:.4f}")
    print(f"Best params: {opt.best_para}")


Next Steps
----------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: User Guide
        :link: user_guide/index
        :link-type: doc

        Deep dive into all features: constraints, memory, stopping conditions,
        initialization strategies, and more.

    .. grid-item-card:: All Algorithms
        :link: user_guide/optimizers/index
        :link-type: doc

        Detailed documentation for all 22 optimization algorithms with
        visualizations and parameter guides.

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc

        Complete code examples for various optimization scenarios from
        simple functions to ML hyperparameter tuning.

    .. grid-item-card:: API Reference
        :link: api_reference/index
        :link-type: doc

        Full API documentation with all parameters and return types.
