=============
Search Spaces
=============

Before running any optimization you need to tell the optimizer which parameters
exist, what values they can take, and what type each parameter is. In GFO a
search space is a Python dictionary mapping parameter names to their domains.
The **type of each domain** is determined by the Python data structure you use:
tuples for continuous ranges, NumPy arrays for discrete grids, and Python lists
for categorical choices. The optimizer adapts its search strategy per dimension
type, so choosing the right data structure matters.


.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Continuous
      :class-card: sd-border-primary gfo-compact

      .. grid:: 2
         :margin: 0
         :padding: 0

         .. grid-item::
            :columns: 5

            A **tuple** for continuous
            ranges without discretization.

         .. grid-item::
            :columns: 7

            .. code-block:: python

               "learning_rate": (0.001, 1.0)

   .. grid-item-card:: Discrete
      :class-card: sd-border-success gfo-compact

      .. grid:: 2
         :margin: 0
         :padding: 0

         .. grid-item::
            :columns: 5

            A **NumPy array** of specific numeric
            values to choose from.

         .. grid-item::
            :columns: 7

            .. code-block:: python

               "max_depth": np.arange(2, 21)

   .. grid-item-card:: Categorical
      :class-card: sd-border-info gfo-compact

      .. grid:: 2
         :margin: 0
         :padding: 0

         .. grid-item::
            :columns: 5

            A **Python list** of unordered choices
            like strings or booleans.

         .. grid-item::
            :columns: 7

            .. code-block:: python

               "kernel": ["linear", "rbf", "poly"]


Quick Example
-------------

All three types can coexist in a single search space:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer

    search_space = {
        "learning_rate": (0.0001, 0.1),           # continuous
        "n_layers": np.arange(1, 6),              # discrete
        "hidden_size": np.arange(32, 256, 32),    # discrete
        "optimizer": ["adam", "sgd", "rmsprop"],   # categorical
        "use_dropout": [True, False],              # categorical (boolean)
    }

    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=100)


Continuous Parameters
---------------------

Use a **tuple** ``(min, max)`` to define a continuous dimension. The optimizer
works directly with float values in this range and is not limited to a fixed
grid of points:

.. code-block:: python

    search_space = {
        "learning_rate": (0.0001, 0.1),
        "temperature": (0.1, 10.0),
        "dropout_rate": (0.0, 0.5),
    }

This is the right choice when you want the optimizer to explore the full
continuous range. For parameters that span several orders of magnitude,
consider using a discrete log-scale grid instead (see below).


Discrete Parameters
-------------------

Use a **NumPy array** to define a grid of specific numeric values. The
optimizer selects from exactly these values, and perturbation-based moves
step through neighboring entries in the array.

**Integer parameters** with ``np.arange``:

.. code-block:: python

    search_space = {
        "n_estimators": np.arange(10, 210, 10),  # 10, 20, ..., 200
        "max_depth": np.arange(2, 21),            # 2, 3, ..., 20
    }

**Fine-grained float grids** with ``np.linspace``:

.. code-block:: python

    search_space = {
        # 200 evenly spaced values from 0.001 to 1.0
        "threshold": np.linspace(0.001, 1.0, 200),
    }

**Log-scale grids** with ``np.logspace``:

.. code-block:: python

    search_space = {
        # 50 values from 0.0001 to 0.1, log-spaced
        "learning_rate": np.logspace(-4, -1, 50),

        # 40 values from 1e-6 to 1e-1
        "regularization": np.logspace(-6, -1, 40),
    }

Log-scale grids are useful for parameters that vary over orders of magnitude.
The array length controls the resolution: more values mean finer granularity.

.. note::

    ``np.linspace`` and ``np.logspace`` create **discrete** grids, not
    continuous ranges. For truly continuous parameters, use a tuple
    ``(min, max)`` instead.


Categorical Parameters
----------------------

Use a **Python list** to define categorical choices. The optimizer treats these
as unordered and uses swap-based moves (jumping to any category) rather than
perturbation-based moves (stepping to neighbors):

.. code-block:: python

    search_space = {
        "optimizer": ["adam", "sgd", "rmsprop", "adagrad"],
        "activation": ["relu", "tanh", "sigmoid"],
        "kernel": ["linear", "rbf", "poly"],
    }

**Booleans** are a special case of categorical:

.. code-block:: python

    search_space = {
        "use_bias": [True, False],
        "normalize": [True, False],
    }


Mixed Search Spaces
-------------------

All dimension types work together in a single search space. The optimizer
handles each dimension according to its type internally:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris

    search_space = {
        "C": (0.01, 100.0),                        # continuous
        "degree": np.arange(2, 6),                  # discrete
        "kernel": ["linear", "rbf", "poly"],        # categorical
        "shrinking": [True, False],                  # categorical (boolean)
    }

    def objective(para):
        X, y = load_iris(return_X_y=True)
        clf = SVC(
            C=para["C"],
            degree=para["degree"],
            kernel=para["kernel"],
            shrinking=para["shrinking"],
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=50)


Search Space Size
-----------------

For discrete and categorical dimensions the total search space size is
the product of all dimension sizes. Continuous dimensions have no fixed
size since they are not discretized.

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

2. **Use appropriate scales**: Log-scale grids (``np.logspace``) for
   parameters that vary over orders of magnitude. Continuous tuples for
   parameters where you want full-range exploration.

3. **Consider dependencies**: If parameters interact, use
   :doc:`constraints <constraints>` to avoid invalid combinations.

4. **Balance discrete dimensions**: Very different dimension sizes can
   bias exploration. A dimension with 1000 values next to one with 2 values
   means the optimizer spends disproportionate effort varying the larger
   dimension while effectively ignoring the smaller one.

.. code-block:: python

    # Good: balanced dimensions
    search_space = {
        "x": np.linspace(0, 1, 50),
        "y": np.linspace(0, 1, 50),
    }

    # Problematic: the optimizer spends most moves varying x
    search_space = {
        "x": np.linspace(0, 1, 1000),  # 1000 values
        "y": np.array([0, 1]),          # 2 values
    }
