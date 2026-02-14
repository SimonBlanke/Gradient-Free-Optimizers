=============
Hill Climbing
=============

Hill Climbing is the simplest local search algorithm. It evaluates neighboring
solutions and moves to the best one, repeating until no improvement is found.


.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/hill_climbing_sphere_function_.gif
            :alt: Hill Climbing on Sphere function

            **Convex function (Sphere)**: Converges quickly and reliably
            to the global optimum.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/hill_climbing_ackley_function_.gif
            :alt: Hill Climbing on Ackley function

            **Multi-modal function (Ackley)**: May get stuck in local
            optima depending on starting point.


Algorithm
---------

At each iteration:

1. Generate ``n_neighbours`` random neighbors within ``epsilon`` distance
2. Evaluate all neighbors
3. Move to the best neighbor if it improves the current score
4. If no improvement, stay in place (may try different neighbors)

.. code-block:: text

    neighbor = current_pos + epsilon * range(dim) * sample(distribution)
    if score(neighbor) > score(current_pos):
        current_pos = neighbor

.. note::

    **Key Insight:** Hill Climbing is a **greedy** algorithm. It always moves toward
    improvement and never accepts worse solutions. This makes it the fastest local
    search method but also the most vulnerable to local optima. Its simplicity makes
    it an ideal building block: most other local search algorithms are modifications
    of Hill Climbing that add escape mechanisms.

.. figure:: /_static/diagrams/hill_climbing_flowchart.svg
    :alt: Hill Climbing algorithm flowchart
    :align: center

    The Hill Climbing loop: generate neighbors, evaluate, and move only
    if an improvement is found.


When to Use
-----------

**Good for:**

- Smooth, unimodal functions
- Fine-tuning from a good starting point
- Fast convergence when the landscape is well-behaved
- Initial exploration before switching to other algorithms

**Not ideal for:**

- Functions with many local optima
- Flat regions (plateaus)
- Unknown landscapes without prior information


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``search_space``
      - dict
      - required
      - Dictionary mapping parameter names to NumPy arrays of values
    * - ``initialize``
      - dict
      - {"grid": 4, "random": 2, "vertices": 4}
      - Initialization strategy
    * - ``constraints``
      - list
      - []
      - List of constraint functions
    * - ``random_state``
      - int
      - None
      - Random seed for reproducibility
    * - ``rand_rest_p``
      - float
      - 0.0
      - Probability of random restart per iteration
    * - ``epsilon``
      - float
      - 0.03
      - Step size as fraction of search space range
    * - ``distribution``
      - str
      - "normal"
      - Step distribution: "normal", "laplace", or "logistic"
    * - ``n_neighbours``
      - int
      - 3
      - Number of neighbors to evaluate per iteration


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def sphere(para):
        return -(para["x"]**2 + para["y"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    opt = HillClimbingOptimizer(
        search_space,
        epsilon=0.05,        # Slightly larger steps
        n_neighbours=5,      # Evaluate more neighbors
    )

    opt.search(sphere, n_iter=500)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Tuning Tips
-----------

**Step Size (epsilon)**

- Larger epsilon: Faster convergence, may overshoot optimum
- Smaller epsilon: Precise convergence, slower exploration

.. code-block:: python

    # Start broad, then fine-tune
    opt1 = HillClimbingOptimizer(search_space, epsilon=0.1)
    opt1.search(objective, n_iter=100)

    opt2 = HillClimbingOptimizer(
        search_space,
        epsilon=0.01,
        initialize={"warm_start": [opt1.best_para]}
    )
    opt2.search(objective, n_iter=100)

**Number of Neighbors**

- More neighbors: Better chance of finding improvements, more evaluations
- Fewer neighbors: Faster iterations, may miss good directions

**Adding Randomness**

Use ``rand_rest_p`` to occasionally jump to random positions:

.. code-block:: python

    opt = HillClimbingOptimizer(
        search_space,
        rand_rest_p=0.1,  # 10% chance of random jump per iteration
    )


Higher-Dimensional Example
--------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def styblinski_tang_4d(para):
        vals = [para["x0"], para["x1"], para["x2"], para["x3"]]
        return -sum(v**4 - 16 * v**2 + 5 * v for v in vals) / 2

    search_space = {
        "x0": np.linspace(-5, 5, 100),
        "x1": np.linspace(-5, 5, 100),
        "x2": np.linspace(-5, 5, 100),
        "x3": np.linspace(-5, 5, 100),
    }

    opt = HillClimbingOptimizer(
        search_space,
        epsilon=0.08,
        n_neighbours=10,
        distribution="logistic",
    )

    opt.search(styblinski_tang_4d, n_iter=2000)
    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: Hill Climbing is purely exploitative. Increase
  ``epsilon`` or ``n_neighbours`` for broader search, but it will never intentionally
  explore distant regions.
- **Computational overhead**: Minimal. The only cost is evaluating ``n_neighbours``
  candidates per iteration.
- **Parameter sensitivity**: Mostly controlled by ``epsilon``. Too small and convergence
  stalls; too large and it oscillates around the optimum.


Related Algorithms
------------------

- :doc:`stochastic_hill_climbing` - Adds probability to accept worse solutions
- :doc:`repulsing_hill_climbing` - Increases step size when stuck
- :doc:`simulated_annealing` - Temperature-based acceptance probability
