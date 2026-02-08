===================
Mixed Search Spaces
===================

Gradient-Free-Optimizers natively supports continuous, discrete, and categorical
dimensions in a single search space. There is no need to encode or transform
parameter types -- the library detects the dimension type from the array you provide
and applies appropriate optimization logic for each type internally.


Dimension Types
---------------

A search space is a Python dictionary mapping parameter names to NumPy arrays
of candidate values. The type of each dimension is determined by the array contents:

**Continuous** -- floating-point values, typically created with ``np.linspace``:

.. code-block:: python

    import numpy as np

    # 200 evenly spaced floats from 0.001 to 1.0
    "learning_rate": np.linspace(0.001, 1.0, 200)

**Discrete** -- integer or evenly spaced numerical values, typically created with ``np.arange``:

.. code-block:: python

    # Integers: 10, 20, 30, ..., 200
    "n_estimators": np.arange(10, 210, 10)

**Categorical** -- strings or other non-numeric values:

.. code-block:: python

    # Categorical choices
    "kernel": np.array(["linear", "rbf", "poly"])

**Boolean** -- a special case of categorical with two values:

.. code-block:: python

    "use_bias": np.array([True, False])


Mixing Types Freely
-------------------

All dimension types can coexist in a single search space dictionary. The optimizer
handles each dimension according to its type, so you can define mixed-type problems
naturally:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import BayesianOptimizer

    # Mixed search space for SVM hyperparameter tuning
    search_space = {
        "C": np.linspace(0.01, 100, 200),           # continuous
        "degree": np.arange(2, 6),                    # discrete
        "kernel": np.array(["linear", "rbf", "poly"]),# categorical
        "shrinking": np.array([True, False]),          # boolean
    }

    def objective(para):
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import load_iris

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
    print(opt.best_para)


Granularity
-----------

The optimizer samples from the values you provide, so the array length controls
the resolution of each dimension. More values mean finer granularity but a larger
search space:

.. code-block:: python

    # Coarse: 10 values
    "x": np.linspace(-10, 10, 10)    # step size = 2.22

    # Fine: 1000 values
    "x": np.linspace(-10, 10, 1000)  # step size = 0.02

For discrete parameters like ``n_estimators``, the step size in ``np.arange``
directly controls the granularity.


Why Native Mixed-Type Support Matters
-------------------------------------

Many optimization libraries require all dimensions to be the same type, or force
you to encode categoricals as integers. GFO uses dimension-type-aware routing
internally: the optimization logic that generates new candidate positions adapts
its strategy per dimension. Continuous dimensions use perturbation-based moves,
while categorical dimensions use swap-based moves. This means:

- No manual encoding or decoding of categorical parameters
- The optimizer uses moves that make sense for each type
- Results are returned using the original values you defined
