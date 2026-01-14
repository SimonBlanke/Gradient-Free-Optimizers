=========================
Constrained Optimization
=========================

This example shows how to use constraints to restrict the search space.


Basic Constraints
-----------------

Constraints are Python functions that return True if the parameters are valid:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # Constraints: x must be positive, and x + y < 5
    constraints = [
        lambda p: p["x"] > 0,
        lambda p: p["x"] + p["y"] < 5,
    ]

    opt = HillClimbingOptimizer(
        search_space,
        constraints=constraints,
    )
    opt.search(objective, n_iter=500)

    print(f"Best: {opt.best_para}")
    # x will be > 0 and x + y < 5


Complex Constraints
-------------------

You can define more complex constraint functions:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import ParticleSwarmOptimizer

    def objective(para):
        return para["x"] * para["y"]

    search_space = {
        "x": np.linspace(0, 10, 100),
        "y": np.linspace(0, 10, 100),
    }

    def budget_constraint(para):
        """Total budget must not exceed 15"""
        return para["x"] + para["y"] <= 15

    def ratio_constraint(para):
        """x should be at most 2x y"""
        return para["x"] <= 2 * para["y"]

    def minimum_constraint(para):
        """Both values must be at least 1"""
        return para["x"] >= 1 and para["y"] >= 1

    constraints = [
        budget_constraint,
        ratio_constraint,
        minimum_constraint,
    ]

    opt = ParticleSwarmOptimizer(search_space, constraints=constraints)
    opt.search(objective, n_iter=300)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


ML Example with Constraints
---------------------------

Constrain hyperparameter relationships:

.. code-block:: python

    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_digits
    from gradient_free_optimizers import BayesianOptimizer

    X, y = load_digits(return_X_y=True)

    def objective(para):
        clf = MLPClassifier(
            hidden_layer_sizes=(para["layer1"], para["layer2"]),
            learning_rate_init=para["lr"],
            alpha=para["alpha"],
            max_iter=200,
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "layer1": np.arange(32, 256, 16),
        "layer2": np.arange(16, 128, 8),
        "lr": np.logspace(-4, -1, 30),
        "alpha": np.logspace(-5, -1, 30),
    }

    # Constraints for neural network architecture
    constraints = [
        # Second layer should be smaller than first (pyramid structure)
        lambda p: p["layer2"] < p["layer1"],
        # Total neurons shouldn't be too large
        lambda p: p["layer1"] + p["layer2"] < 300,
    ]

    opt = BayesianOptimizer(search_space, constraints=constraints)
    opt.search(objective, n_iter=30)

    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Architecture: ({opt.best_para['layer1']}, {opt.best_para['layer2']})")


.. note::

    When constraints are violated, the optimizer automatically retries with
    different positions until a valid configuration is found. This may slow
    down initialization if constraints are very restrictive.
