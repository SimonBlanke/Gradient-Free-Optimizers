===========
Constraints
===========

Constraints restrict which parameter combinations are valid. The optimizer
automatically avoids invalid regions of the search space.


Defining Constraints
--------------------

Constraints are Python functions that take a parameter dictionary and return
``True`` if the parameters are valid, ``False`` otherwise.

.. code-block:: python

    # Simple constraint: x must be positive
    constraint = lambda params: params["x"] > 0

    # Multiple constraints as a list
    constraints = [
        lambda p: p["x"] > 0,           # x must be positive
        lambda p: p["x"] + p["y"] < 10, # sum must be less than 10
    ]


Using Constraints
-----------------

Pass constraints to the optimizer constructor:

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    constraints = [
        lambda p: p["x"] > 0,
        lambda p: p["y"] > p["x"],
    ]

    opt = HillClimbingOptimizer(
        search_space,
        constraints=constraints,
    )

    opt.search(objective, n_iter=500)


Common Constraint Patterns
--------------------------

**Range constraints:**

.. code-block:: python

    constraints = [
        lambda p: 0 < p["x"] < 5,          # x between 0 and 5
        lambda p: p["y"] >= 1,              # y at least 1
    ]

**Relationship constraints:**

.. code-block:: python

    constraints = [
        lambda p: p["x"] < p["y"],          # x must be less than y
        lambda p: p["x"] + p["y"] <= 10,    # Sum constraint
        lambda p: p["x"] * p["y"] > 0,      # Product constraint
    ]

**Categorical logic:**

.. code-block:: python

    constraints = [
        # If using adam, learning_rate must be < 0.01
        lambda p: p["optimizer"] != "adam" or p["learning_rate"] < 0.01,
    ]

**Complex constraints:**

.. code-block:: python

    def valid_architecture(params):
        """Each layer must be smaller than the previous"""
        return (params["layer1"] > params["layer2"] >
                params["layer3"])

    def budget_constraint(params):
        """Total compute must be within budget"""
        compute = params["n_layers"] * params["hidden_size"]
        return compute <= 10000

    constraints = [valid_architecture, budget_constraint]


ML Hyperparameter Example
-------------------------

.. code-block:: python

    search_space = {
        "n_estimators": np.arange(10, 200, 10),
        "max_depth": np.arange(2, 20),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 10),
    }

    constraints = [
        # min_samples_split must be greater than min_samples_leaf
        lambda p: p["min_samples_split"] > p["min_samples_leaf"],

        # Avoid very deep trees with many estimators (too slow)
        lambda p: p["max_depth"] * p["n_estimators"] < 2000,
    ]


How Constraints Work
--------------------

When the optimizer proposes a new position:

1. Convert position to parameters
2. Check all constraint functions
3. If any returns ``False``, reject and try another position
4. Repeat until valid position found

.. warning::

    Very restrictive constraints can slow down initialization and search,
    as the optimizer may need many retries to find valid positions.


Best Practices
--------------

1. **Keep constraints simple**: Complex constraints are harder to satisfy
2. **Test constraints**: Verify valid combinations exist before running
3. **Use search space narrowing first**: If possible, restrict the search
   space instead of adding constraints
4. **Combine wisely**: Too many constraints may make optimization infeasible

.. code-block:: python

    # Instead of this:
    search_space = {"x": np.linspace(-100, 100, 1000)}
    constraints = [lambda p: 0 < p["x"] < 10]

    # Do this:
    search_space = {"x": np.linspace(0.1, 10, 100)}
    # No constraints needed!
