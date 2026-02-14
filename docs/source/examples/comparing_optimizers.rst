=====================
Comparing Optimizers
=====================

This example shows how to benchmark different optimization algorithms on the
same problem.


Basic Comparison
----------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import (
        HillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        RandomSearchOptimizer,
        ParticleSwarmOptimizer,
        BayesianOptimizer,
    )

    def rastrigin(para):
        x, y = para["x"], para["y"]
        A = 10
        return -(A * 2 + (x**2 - A * np.cos(2 * np.pi * x))
                      + (y**2 - A * np.cos(2 * np.pi * y)))

    search_space = {
        "x": np.linspace(-5.12, 5.12, 200),
        "y": np.linspace(-5.12, 5.12, 200),
    }

    optimizers = [
        ("Hill Climbing", HillClimbingOptimizer),
        ("Simulated Annealing", SimulatedAnnealingOptimizer),
        ("Random Search", RandomSearchOptimizer),
        ("Particle Swarm", ParticleSwarmOptimizer),
        ("Bayesian", BayesianOptimizer),
    ]

    n_iter = 200
    results = []

    for name, OptimizerClass in optimizers:
        opt = OptimizerClass(search_space, random_state=42)
        opt.search(rastrigin, n_iter=n_iter)
        results.append((name, opt.best_score, opt.best_para))
        print(f"{name:25s}: score = {opt.best_score:.4f}")

    # Find best
    best_name, best_score, best_para = max(results, key=lambda x: x[1])
    print(f"\nBest optimizer: {best_name} with score {best_score:.4f}")


Multiple Runs for Statistics
----------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import (
        HillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        BayesianOptimizer,
    )

    def sphere(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    optimizers = [
        ("Hill Climbing", HillClimbingOptimizer, {}),
        ("Simulated Annealing", SimulatedAnnealingOptimizer, {"annealing_rate": 0.97}),
        ("Bayesian", BayesianOptimizer, {}),
    ]

    n_runs = 10
    n_iter = 100

    print(f"{'Optimizer':<25} {'Mean':>10} {'Std':>10} {'Best':>10}")
    print("-" * 60)

    for name, OptimizerClass, kwargs in optimizers:
        scores = []
        for seed in range(n_runs):
            opt = OptimizerClass(search_space, random_state=seed, **kwargs)
            opt.search(sphere, n_iter=n_iter)
            scores.append(opt.best_score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        best_score = max(scores)
        print(f"{name:<25} {mean_score:>10.4f} {std_score:>10.4f} {best_score:>10.4f}")


Convergence Curves
------------------

Track optimization progress over iterations:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import (
        HillClimbingOptimizer,
        BayesianOptimizer,
        ParticleSwarmOptimizer,
    )

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    def track_convergence(OptimizerClass, name, **kwargs):
        """Track best score across iterations"""
        opt = OptimizerClass(search_space, random_state=42, **kwargs)
        opt.search(objective, n_iter=100, verbosity=False)

        scores = opt.search_data["score"].values
        best_scores = np.maximum.accumulate(scores)

        return name, best_scores

    results = [
        track_convergence(HillClimbingOptimizer, "Hill Climbing"),
        track_convergence(BayesianOptimizer, "Bayesian"),
        track_convergence(ParticleSwarmOptimizer, "PSO", population=10),
    ]

    # Print convergence at key iterations
    print(f"{'Optimizer':<20} {'@10':>10} {'@25':>10} {'@50':>10} {'@100':>10}")
    print("-" * 60)
    for name, scores in results:
        print(f"{name:<20} {scores[9]:>10.4f} {scores[24]:>10.4f} "
              f"{scores[49]:>10.4f} {scores[99]:>10.4f}")


ML Hyperparameter Comparison
----------------------------

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    from gradient_free_optimizers import (
        RandomSearchOptimizer,
        BayesianOptimizer,
        TreeStructuredParzenEstimators,
    )

    X, y = load_iris(return_X_y=True)

    def objective(para):
        clf = RandomForestClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=5).mean()

    search_space = {
        "n_estimators": np.arange(10, 200, 10),
        "max_depth": np.arange(2, 20),
    }

    optimizers = [
        ("Random Search", RandomSearchOptimizer, {}),
        ("Bayesian", BayesianOptimizer, {}),
        ("TPE", TreeStructuredParzenEstimators, {}),
    ]

    n_iter = 30
    print(f"\nComparing optimizers ({n_iter} iterations):\n")

    for name, OptimizerClass, kwargs in optimizers:
        opt = OptimizerClass(search_space, random_state=42, **kwargs)
        opt.search(objective, n_iter=n_iter)
        print(f"{name:20s}: accuracy = {opt.best_score:.4f}, "
              f"params = {opt.best_para}")
