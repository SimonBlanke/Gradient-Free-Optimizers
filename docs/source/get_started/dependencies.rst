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
- **SciPy** -- used internally by some optimizers (e.g. distance calculations)
- **tqdm** -- progress bar display during ``search()``

That's it. No heavy frameworks, no compiled C extensions beyond what NumPy
and SciPy already provide, no external optimization backends.


Optional Dependencies
---------------------

For surrogate-model-based optimizers (``BayesianOptimizer``, ``ForestOptimizer``,
``TreeStructuredParzenEstimators``), you can optionally install scikit-learn to
use sklearn-based surrogate models:

.. code-block:: bash

    pip install gradient-free-optimizers[sklearn]

This adds:

- **scikit-learn** -- provides alternative GP, Random Forest, and KDE implementations
  for surrogate models

However, sklearn is **not required**. GFO ships its own Gaussian Process, Random
Forest, and KDE implementations, so all 22 optimizers work out of the box without
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

**No conflicts.** NumPy, pandas, SciPy, and tqdm are among the most widely used
Python packages. Adding GFO to an existing project is unlikely to introduce
version conflicts.


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
