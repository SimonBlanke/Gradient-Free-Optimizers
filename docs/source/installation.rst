.. _installation:

============
Installation
============

Gradient-Free-Optimizers can be installed via pip and supports Python 3.10+.

----

Installing Gradient-Free-Optimizers
====================================

Basic Installation
------------------

Install GFO from PyPI using pip:

.. code-block:: bash

    pip install gradient-free-optimizers

This installs GFO with its minimal dependencies (only NumPy), which is sufficient
for all core optimization algorithms.


Installation with Extras
-------------------------

For additional functionality, you can install optional extras:

.. tab-set::

    .. tab-item:: Progress Bars

        .. code-block:: bash

            pip install gradient-free-optimizers[progress]

        Adds ``tqdm`` for progress bars during optimization.

    .. tab-item:: SMBO Algorithms

        .. code-block:: bash

            pip install gradient-free-optimizers[sklearn]

        Adds ``scikit-learn`` for surrogate models used in Bayesian Optimization,
        TPE, Forest Optimizer, and Ensemble Optimizer.

    .. tab-item:: Full Installation

        .. code-block:: bash

            pip install gradient-free-optimizers[full]

        Installs all optional dependencies (tqdm + scikit-learn).


Development Installation
------------------------

To install GFO for development (from source):

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/SimonBlanke/Gradient-Free-Optimizers.git
    cd Gradient-Free-Optimizers

    # Install in development mode with test dependencies
    pip install -e ".[test]"

    # Or install with all development dependencies
    pip install -e ".[test,docs]"

    # Run tests to verify installation
    pytest tests/


----

Dependencies
============

Core Dependencies
-----------------

Gradient-Free-Optimizers requires the following packages (automatically installed):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``numpy >= 1.18``
     - Numerical operations, array handling, and search space definition
   * - ``pandas >= 1.0``
     - Search data storage and manipulation

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``tqdm >= 4.48`` (extra: ``progress``)
     - Progress bars during optimization
   * - ``scikit-learn >= 0.23`` (extra: ``sklearn``)
     - Surrogate models for SMBO algorithms (Bayesian, TPE, Forest, Ensemble)


----

Verifying Installation
======================

After installation, verify that GFO works correctly:

.. code-block:: python

    import gradient_free_optimizers
    import numpy as np

    # Check version
    print(f"GFO version: {gradient_free_optimizers.__version__}")

    # Run a quick optimization
    from gradient_free_optimizers import HillClimbingOptimizer

    def sphere(params):
        return -(params["x"]**2 + params["y"]**2)

    search_space = {
        "x": np.linspace(-5, 5, 50),
        "y": np.linspace(-5, 5, 50),
    }

    opt = HillClimbingOptimizer(search_space)
    opt.search(sphere, n_iter=100)

    print(f"Best score: {opt.best_score}")
    print(f"Best params: {opt.best_para}")

Expected output:

.. code-block:: text

    GFO version: X.X.X
    Best score: -0.04  (approximately)
    Best params: {'x': 0.2, 'y': 0.0}  (approximately)


----

Troubleshooting Installation
=============================

Python Version Issues
---------------------

GFO requires Python 3.10 or newer. Check your Python version:

.. code-block:: bash

    python --version

If you have multiple Python versions, use:

.. code-block:: bash

    python3.10 -m pip install gradient-free-optimizers


NumPy Compatibility
-------------------

GFO supports both NumPy 1.x and 2.x. If you encounter issues:

.. code-block:: bash

    # Install with specific NumPy version
    pip install "numpy<2.0" gradient-free-optimizers  # For NumPy 1.x
    pip install "numpy>=2.0" gradient-free-optimizers  # For NumPy 2.x


Dependency Conflicts
--------------------

If you encounter dependency conflicts:

.. code-block:: bash

    # Create a fresh virtual environment
    python -m venv gfo_env
    source gfo_env/bin/activate  # On Windows: gfo_env\Scripts\activate
    pip install gradient-free-optimizers


Permission Errors
-----------------

If you get permission errors during installation:

.. code-block:: bash

    # Install for current user only (recommended)
    pip install --user gradient-free-optimizers

    # Or use a virtual environment (better practice)
    python -m venv myenv
    source myenv/bin/activate
    pip install gradient-free-optimizers


Missing SMBO Dependencies
--------------------------

If you try to use Bayesian Optimization or other SMBO algorithms without
scikit-learn installed, you'll see an import error. Install with:

.. code-block:: bash

    pip install gradient-free-optimizers[sklearn]


----

Installation in Special Environments
=====================================

Jupyter Notebooks
-----------------

Install directly in a notebook cell:

.. code-block:: python

    !pip install gradient-free-optimizers

Or with extras:

.. code-block:: python

    !pip install gradient-free-optimizers[full]


Google Colab
------------

GFO works out of the box in Google Colab:

.. code-block:: python

    !pip install gradient-free-optimizers
    import gradient_free_optimizers


Docker
------

Add to your ``Dockerfile``:

.. code-block:: dockerfile

    FROM python:3.11-slim
    RUN pip install gradient-free-optimizers[full]


Conda Environments
------------------

While GFO is not on conda-forge yet, you can install via pip in a conda environment:

.. code-block:: bash

    conda create -n gfo python=3.11
    conda activate gfo
    pip install gradient-free-optimizers


----

Using Older Versions
====================

If you need to use an older version of GFO, you can install a specific version:

.. code-block:: bash

    pip install gradient-free-optimizers==1.5.0

Documentation for older versions is available at:
`Legacy Documentation (v1.x) <https://simonblanke.github.io/gradient-free-optimizers-documentation/1.5/>`_

----

Next Steps
==========

- **Quick Start**: Head to :doc:`get_started` for your first optimization
- **User Guide**: Learn about :doc:`user_guide/search_spaces` and :doc:`user_guide/optimizers/index`
- **Examples**: See practical usage in :doc:`examples`
