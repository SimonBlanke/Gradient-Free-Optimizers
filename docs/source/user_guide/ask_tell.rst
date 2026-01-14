===================
Ask-Tell Interface
===================

The ask-tell interface provides manual control over the optimization loop,
useful for distributed computing, custom logging, or integration with
external systems.


Basic Usage
-----------

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_space)
    opt.setup_search(objective, n_iter=100)  # Setup, don't run

    for i in range(100):
        params = opt.ask()           # Get next parameters
        score = objective(params)    # Evaluate
        opt.tell(params, score)      # Report result

    print(opt.best_para)


Methods
-------

**setup_search()**

Initialize the optimization without running it:

.. code-block:: python

    opt.setup_search(
        objective_function,
        n_iter,
        max_time=None,
        max_score=None,
        memory=True,
        memory_warm_start=None,
        verbosity=[],
    )

**ask()**

Get the next parameters to evaluate:

.. code-block:: python

    params = opt.ask()  # Returns dict: {"x": 0.5, "y": 1.2, ...}

**tell()**

Report an evaluation result:

.. code-block:: python

    opt.tell(params, score)


When to Use Ask-Tell
--------------------

**Custom stopping conditions:**

.. code-block:: python

    opt.setup_search(objective, n_iter=10000)

    while True:
        params = opt.ask()
        score = objective(params)
        opt.tell(params, score)

        if score > 0.99:
            print("Target reached!")
            break

        if time.time() - start > 3600:
            print("Time limit reached!")
            break

**Custom logging:**

.. code-block:: python

    for i in range(100):
        params = opt.ask()
        score = objective(params)
        opt.tell(params, score)

        # Custom logging
        log.info(f"Iter {i}: params={params}, score={score:.4f}")
        mlflow.log_metrics({"score": score}, step=i)

**Distributed evaluation:**

.. code-block:: python

    opt.setup_search(objective, n_iter=100)

    # Collect batch of evaluations
    batch = []
    for _ in range(10):
        params = opt.ask()
        batch.append(params)

    # Evaluate in parallel (pseudo-code)
    scores = parallel_evaluate(batch)

    # Report results
    for params, score in zip(batch, scores):
        opt.tell(params, score)

**External systems:**

.. code-block:: python

    opt.setup_search(objective, n_iter=100)

    for i in range(100):
        params = opt.ask()

        # Send to external system
        job_id = external_system.submit(params)

        # Wait for result
        score = external_system.get_result(job_id)

        opt.tell(params, score)


Complete Example
----------------

.. code-block:: python

    import numpy as np
    import time
    from gradient_free_optimizers import BayesianOptimizer

    def objective(params):
        return -(params["x"]**2 + params["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = BayesianOptimizer(search_space)
    opt.setup_search(objective, n_iter=500)

    start_time = time.time()
    iteration = 0

    while iteration < 500:
        params = opt.ask()
        score = objective(params)
        opt.tell(params, score)
        iteration += 1

        # Log every 50 iterations
        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration}: best={opt.best_score:.4f}, "
                  f"time={elapsed:.1f}s")

        # Early stopping
        if opt.best_score > -0.01:
            print(f"Converged at iteration {iteration}")
            break

    print(f"\nFinal: {opt.best_para}")
    print(f"Score: {opt.best_score}")
