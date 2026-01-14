=============
Search Spaces
=============

Search spaces define what parameters the optimizer can explore. In GFO,
search spaces are simple dictionaries mapping parameter names to NumPy
arrays of possible values.


Basic Definition
----------------

.. code-block:: python

    import numpy as np

    search_space = {
        "parameter_name": np.array([value1, value2, value3, ...]),
    }

Each key is a parameter name (string), and each value is a NumPy array
containing all possible values for that parameter.


Continuous Parameters
---------------------

For continuous parameters, use ``np.linspace`` or ``np.arange``:

.. code-block:: python

    search_space = {
        # 100 evenly spaced values from 0.001 to 1.0
        "learning_rate": np.linspace(0.001, 1.0, 100),

        # Values from 0 to 10 with step 0.1
        "threshold": np.arange(0, 10, 0.1),
    }

The **granularity** of your array determines how precisely the optimizer
can tune the parameter.


Log-Scale Parameters
--------------------

Many hyperparameters (like learning rates) work better on a log scale:

.. code-block:: python

    search_space = {
        # 50 values from 0.0001 to 0.1, log-spaced
        "learning_rate": np.logspace(-4, -1, 50),

        # From 1e-6 to 1e-1
        "regularization": np.logspace(-6, -1, 40),
    }


Discrete Integer Parameters
---------------------------

For integer parameters, use ``np.arange``:

.. code-block:: python

    search_space = {
        # Integers 10, 20, 30, ..., 200
        "n_estimators": np.arange(10, 210, 10),

        # Integers 2, 3, 4, ..., 20
        "max_depth": np.arange(2, 21),
    }


Categorical Parameters
----------------------

For categorical parameters, use arrays of strings:

.. code-block:: python

    search_space = {
        "optimizer": np.array(["adam", "sgd", "rmsprop", "adagrad"]),
        "activation": np.array(["relu", "tanh", "sigmoid"]),
        "kernel": np.array(["linear", "rbf", "poly"]),
    }


Boolean Parameters
------------------

Use arrays with True/False:

.. code-block:: python

    search_space = {
        "use_bias": np.array([True, False]),
        "normalize": np.array([True, False]),
    }


Mixed Search Spaces
-------------------

Combine different parameter types:

.. code-block:: python

    search_space = {
        # Continuous
        "learning_rate": np.logspace(-4, -1, 30),

        # Discrete
        "n_layers": np.arange(1, 6),
        "hidden_size": np.arange(32, 256, 32),

        # Categorical
        "optimizer": np.array(["adam", "sgd"]),
        "activation": np.array(["relu", "tanh"]),

        # Boolean
        "use_dropout": np.array([True, False]),
    }


Search Space Size
-----------------

The total search space size is the product of all dimension sizes:

.. code-block:: python

    search_space = {
        "x": np.linspace(-10, 10, 100),  # 100 values
        "y": np.linspace(-10, 10, 100),  # 100 values
        "z": np.arange(1, 11),           # 10 values
    }
    # Total: 100 * 100 * 10 = 100,000 possible combinations

Larger search spaces require more iterations to explore effectively.


Best Practices
--------------

1. **Start coarse, refine later**: Begin with fewer values and increase
   granularity once you find promising regions.

2. **Use appropriate scales**: Log-scale for parameters that vary over
   orders of magnitude.

3. **Consider dependencies**: If parameters interact, consider using
   constraints to avoid invalid combinations.

4. **Balance dimensions**: Very different dimension sizes can cause
   some parameters to be under-explored.

.. code-block:: python

    # Good: balanced dimensions
    search_space = {
        "x": np.linspace(0, 1, 50),
        "y": np.linspace(0, 1, 50),
    }

    # May cause issues: imbalanced dimensions
    search_space = {
        "x": np.linspace(0, 1, 1000),  # Very fine
        "y": np.array([0, 1]),          # Very coarse
    }
