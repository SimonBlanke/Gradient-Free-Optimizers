======================
Minimal Dependencies
======================

Gradient-Free-Optimizers is designed to have a small dependency footprint.
The core library requires only packages that are already present in virtually
every Python data science environment.


Core Dependencies
-----------------

The following packages are required:

- **NumPy** -- array operations and search space representation
- **pandas** -- results storage via ``opt.search_data`` DataFrame

That's it. No heavy frameworks and no external optimization backends are
required for the standard optimizer API.


Optional Dependencies
---------------------

Optional extras enable additional integrations:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Install command
     - Adds
   * - ``progress``
     - ``pip install gradient-free-optimizers[progress]``
     - ``tqdm`` progress bars during ``search()``
   * - ``scipy``
     - ``pip install gradient-free-optimizers[scipy]``
     - SciPy distribution-backed search-space dimensions
   * - ``sklearn``
     - ``pip install gradient-free-optimizers[sklearn]``
     - scikit-learn surrogate models for SMBO optimizers
   * - ``full``
     - ``pip install gradient-free-optimizers[full]``
     - all optional dependencies

For surrogate-model-based optimizers (``BayesianOptimizer``, ``ForestOptimizer``,
``TreeStructuredParzenEstimators``), scikit-learn is optional. Install it when
you want to use sklearn-based surrogate models:

.. code-block:: bash

    pip install gradient-free-optimizers[sklearn]

However, sklearn is **not required**. GFO ships its own Gaussian Process, Random
Forest, and KDE implementations, so all 23 optimizers work out of the box without
scikit-learn installed.


Why This Matters
----------------

**Fast installation.** ``pip install gradient-free-optimizers`` completes in seconds.
There are no large compiled dependencies to build or download beyond the standard
scientific Python stack.

**Small footprint.** The flat dependency tree means no transitive dependency
surprises. You know exactly what gets installed.

**Works everywhere.** The minimal requirements mean GFO runs reliably in constrained
environments:

- Docker containers with slim base images
- CI/CD pipelines where install time matters
- Serverless functions with package size limits
- Air-gapped environments with limited package availability

**No conflicts.** NumPy and pandas are among the most widely used Python
packages. SciPy, tqdm, and scikit-learn stay optional, so downstream libraries
can avoid pulling them in unless they need those features.


Built-in Surrogate Models
-------------------------

A key design decision: GFO provides its own implementations of the surrogate models
used by sequential model-based optimizers. This means ``BayesianOptimizer`` works
immediately after installation, without scikit-learn:

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer
    import numpy as np

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    def objective(para):
        return -(para["x"] ** 2 + para["y"] ** 2)

    # Works without sklearn -- uses GFO's built-in GP
    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=50)
    print(opt.best_para)

If sklearn is available, you can pass an sklearn estimator via the ``gpr``
parameter for ``BayesianOptimizer``, or configure the tree-based surrogates via
``tree_regressor`` and ``tree_para`` in ``ForestOptimizer``.
