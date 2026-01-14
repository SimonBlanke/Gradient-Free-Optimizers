========================
Local Search Algorithms
========================

Local search algorithms explore the neighborhood of the current best solution,
making small adjustments to find improvements. They are fast and efficient but
may get stuck in local optima for complex landscapes.


Overview
--------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Algorithm
      - Description
    * - :doc:`hill_climbing`
      - Evaluates neighbors and moves to the best one. Simple and effective.
    * - :doc:`stochastic_hill_climbing`
      - Accepts worse solutions with some probability to escape local optima.
    * - :doc:`repulsing_hill_climbing`
      - Increases step size when stuck to explore further.
    * - :doc:`simulated_annealing`
      - Temperature-based acceptance of worse solutions that decreases over time.
    * - :doc:`downhill_simplex`
      - Geometric method using a simplex of n+1 points.


When to Use Local Search
------------------------

**Good for:**

- Smooth, unimodal objective functions
- Fine-tuning solutions from a good starting point
- Fast iterations when overhead matters
- Problems where function evaluations are very cheap

**Not ideal for:**

- Multi-modal functions with many local optima
- Unknown landscapes without good starting points
- Problems requiring guaranteed global convergence


Common Parameters
-----------------

All local search algorithms share these parameters:

.. list-table::
    :header-rows: 1
    :widths: 20 15 65

    * - Parameter
      - Default
      - Description
    * - ``epsilon``
      - 0.03
      - Step size as fraction of search space range
    * - ``distribution``
      - "normal"
      - How steps are sampled: "normal", "laplace", or "logistic"
    * - ``n_neighbours``
      - 3
      - Number of neighbors to evaluate per iteration


Step Size (epsilon)
^^^^^^^^^^^^^^^^^^^

The ``epsilon`` parameter controls how far the algorithm looks for neighbors:

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    # Small steps for fine-tuning
    opt = HillClimbingOptimizer(search_space, epsilon=0.01)

    # Larger steps for broader exploration
    opt = HillClimbingOptimizer(search_space, epsilon=0.1)

.. tip::

    Start with the default ``epsilon=0.03`` and adjust based on results:

    - Converging too slowly? Increase epsilon
    - Jumping over good solutions? Decrease epsilon


Distribution
^^^^^^^^^^^^

The ``distribution`` parameter affects how step sizes are sampled:

- **"normal"** (default): Most steps are small, occasional larger steps
- **"laplace"**: Sharper peak, more small steps
- **"logistic"**: Heavier tails, more large steps

.. code-block:: python

    # More aggressive exploration
    opt = HillClimbingOptimizer(search_space, distribution="logistic")


Visualization
-------------

See how local search algorithms behave on test functions:

.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/hill_climbing_sphere_function_.gif
            :alt: Hill Climbing on Sphere function

            Hill Climbing on a convex (Sphere) function - converges quickly
            to the optimum.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/hill_climbing_ackley_function_.gif
            :alt: Hill Climbing on Ackley function

            Hill Climbing on a multi-modal (Ackley) function - may get stuck
            in local optima.


Algorithms
----------

.. toctree::
    :maxdepth: 1

    hill_climbing
    stochastic_hill_climbing
    repulsing_hill_climbing
    simulated_annealing
    downhill_simplex
