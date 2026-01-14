===================
Ask-Tell Interface
===================

The ask-tell interface gives you manual control over the optimization loop,
useful for distributed computing, custom logging, or integration with external
systems.


Basic Ask-Tell
--------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # Create optimizer and set up (but don't run) search
    opt = HillClimbingOptimizer(search_space)
    opt.setup_search(objective, n_iter=100)

    # Manual optimization loop
    for i in range(100):
        # Get next parameters to evaluate
        params = opt.ask()

        # Evaluate (this could happen anywhere)
        score = objective(params)

        # Report result back
        opt.tell(params, score)

        # Custom logging
        if i % 20 == 0:
            print(f"Iteration {i}: best = {opt.best_score:.4f}")

    print(f"\nFinal: {opt.best_para}, Score: {opt.best_score}")


Custom Stopping Conditions
--------------------------

.. code-block:: python

    import numpy as np
    import time
    from gradient_free_optimizers import BayesianOptimizer

    def expensive_objective(para):
        # Simulate expensive computation
        time.sleep(0.1)
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = BayesianOptimizer(search_space)
    opt.setup_search(expensive_objective, n_iter=1000)

    start_time = time.time()
    target_score = -0.1
    max_time = 60  # seconds

    iteration = 0
    while True:
        params = opt.ask()
        score = expensive_objective(params)
        opt.tell(params, score)
        iteration += 1

        # Custom stopping conditions
        if score >= target_score:
            print(f"Target reached at iteration {iteration}!")
            break

        if time.time() - start_time > max_time:
            print(f"Time limit reached after {iteration} iterations")
            break

        if iteration >= 500:
            print("Maximum iterations reached")
            break

    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


Batch Evaluation
----------------

For parallelizable objective functions:

.. code-block:: python

    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from gradient_free_optimizers import ParticleSwarmOptimizer

    def objective(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = ParticleSwarmOptimizer(search_space, population=10)
    opt.setup_search(objective, n_iter=100)

    # Note: For true parallelism with PSO, you'd need to modify
    # the approach since PSO updates are sequential within generations

    for generation in range(10):
        # Collect a batch of evaluations
        batch_params = []
        batch_scores = []

        for _ in range(10):  # Batch size
            params = opt.ask()
            score = objective(params)  # Could be parallelized
            opt.tell(params, score)
            batch_params.append(params)
            batch_scores.append(score)

        best_in_batch = max(batch_scores)
        print(f"Generation {generation}: best in batch = {best_in_batch:.4f}")

    print(f"\nFinal best: {opt.best_score}")


External Evaluation
-------------------

When the objective function runs on a different system:

.. code-block:: python

    import numpy as np
    import json
    from gradient_free_optimizers import BayesianOptimizer

    def simulate_external_evaluation(params_json):
        """Simulate sending params to external system and getting result"""
        params = json.loads(params_json)
        # In reality, this would call an API, run a simulation, etc.
        return -(params["x"]**2 + params["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = BayesianOptimizer(search_space)
    opt.setup_search(lambda p: 0, n_iter=50)  # Dummy objective

    for i in range(50):
        # Get parameters
        params = opt.ask()

        # Convert to JSON for external system
        params_json = json.dumps({k: float(v) for k, v in params.items()})

        # External evaluation
        score = simulate_external_evaluation(params_json)

        # Report back
        opt.tell(params, score)

    print(f"Best: {opt.best_para}")
