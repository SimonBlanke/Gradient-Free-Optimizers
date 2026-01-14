==============
Initialization
==============

Initialization controls where the optimizer starts its search. Good
initialization can significantly improve optimization efficiency.


Default Initialization
----------------------

By default, GFO uses a mix of strategies:

.. code-block:: python

    initialize = {
        "grid": 4,      # 4 grid-spaced positions
        "random": 2,    # 2 random positions
        "vertices": 4,  # 4 corner/edge positions
    }

This provides a balance of systematic and random coverage.


Initialization Options
----------------------

**Grid initialization:**

Generates positions on a regular grid across the search space.

.. code-block:: python

    initialize = {"grid": 10}  # 10 grid positions


**Random initialization:**

Generates random positions.

.. code-block:: python

    initialize = {"random": 20}  # 20 random positions


**Vertices initialization:**

Generates positions at corners and edges of the search space.

.. code-block:: python

    initialize = {"vertices": 8}  # 8 vertex positions


**Warm start initialization:**

Start from specific known positions.

.. code-block:: python

    initialize = {
        "warm_start": [
            {"x": 0.5, "y": 1.0},   # First starting point
            {"x": -0.5, "y": 2.0},  # Second starting point
        ]
    }


Combining Strategies
--------------------

Mix multiple strategies:

.. code-block:: python

    initialize = {
        "grid": 4,
        "random": 4,
        "vertices": 2,
        "warm_start": [
            {"x": 0.0, "y": 0.0},  # Known good starting point
        ],
    }


Strategy Selection
------------------

**Use grid when:**

- You want systematic coverage
- The search space is small
- You don't have prior knowledge

**Use random when:**

- The search space is large
- You want diverse starting points
- Using population-based algorithms

**Use vertices when:**

- Optima might be at boundaries
- You want to test extreme values

**Use warm_start when:**

- You have prior knowledge of good regions
- Continuing from a previous optimization
- Fine-tuning around known solutions


Examples
--------

**Exploration-focused:**

.. code-block:: python

    # Many diverse starting points
    opt = HillClimbingOptimizer(
        search_space,
        initialize={"random": 50}
    )

**Exploitation-focused:**

.. code-block:: python

    # Start from known good region
    opt = HillClimbingOptimizer(
        search_space,
        initialize={
            "warm_start": [previous_best_params],
            "random": 2,  # Plus some exploration
        }
    )

**Population-based:**

.. code-block:: python

    # Ensure diverse initial population
    opt = ParticleSwarmOptimizer(
        search_space,
        population=20,
        initialize={
            "grid": 5,
            "random": 10,
            "vertices": 5,
        }
    )


Notes
-----

- Initial positions are evaluated first before the main optimization loop
- For population-based algorithms, initial positions are distributed across individuals
- Constraints are respected during initialization (invalid positions are rejected)
