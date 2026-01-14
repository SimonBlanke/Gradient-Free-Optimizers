.. _troubleshooting_installation:

=====================
Installation Problems
=====================

Solutions for common installation and import issues.

----

Import Errors
=============

"No module named 'gradient_free_optimizers'"
---------------------------------------------

**Problem**: Python can't find the installed package.

**Solutions**:

1. Verify installation:

   .. code-block:: bash

       pip list | grep gradient-free-optimizers

2. Check you're using the correct Python environment:

   .. code-block:: bash

       which python
       python --version

3. Reinstall in the active environment:

   .. code-block:: bash

       pip install --upgrade gradient-free-optimizers

4. If using virtual environments, ensure it's activated:

   .. code-block:: bash

       # Linux/Mac
       source venv/bin/activate

       # Windows
       venv\Scripts\activate


"No module named 'sklearn'" or "No module named 'tqdm'"
--------------------------------------------------------

**Problem**: Optional dependencies missing for SMBO algorithms or progress bars.

**Solution**: Install extras:

.. code-block:: bash

    # For SMBO algorithms (Bayesian, TPE, Forest, Ensemble)
    pip install gradient-free-optimizers[sklearn]

    # For progress bars
    pip install gradient-free-optimizers[progress]

    # Or install everything
    pip install gradient-free-optimizers[full]


----

Version Conflicts
=================

NumPy Version Mismatch
----------------------

**Problem**: Conflicts between NumPy 1.x and 2.x.

**Solution**: GFO supports both. Force a specific version:

.. code-block:: bash

    # For NumPy 1.x
    pip install "numpy<2.0" gradient-free-optimizers

    # For NumPy 2.x
    pip install "numpy>=2.0" gradient-free-optimizers


Python Version Too Old
----------------------

**Problem**: GFO requires Python 3.10+.

**Error message**: ``Requires-Python >=3.10``

**Solution**: Upgrade Python or use an older GFO version:

.. code-block:: bash

    # Check Python version
    python --version

    # Install compatible GFO version for older Python
    # (Python 3.7-3.9 supported in GFO < 2.0)
    pip install "gradient-free-optimizers<2.0"


Pandas Version Issues
---------------------

**Problem**: Pandas 3.x compatibility issues.

**Solution**: Use Pandas 2.x:

.. code-block:: bash

    pip install "pandas<3.0" gradient-free-optimizers


----

Permission Errors
=================

"Permission denied" During Installation
----------------------------------------

**Problem**: Insufficient permissions to install system-wide.

**Solution**: Install for current user:

.. code-block:: bash

    pip install --user gradient-free-optimizers


Or use a virtual environment (recommended):

.. code-block:: bash

    python -m venv gfo_env
    source gfo_env/bin/activate
    pip install gradient-free-optimizers


----

Platform-Specific Issues
========================

Windows: "error: Microsoft Visual C++ 14.0 is required"
--------------------------------------------------------

**Problem**: Missing C++ compiler for dependencies.

**Solution**: Install Visual C++ Build Tools or use pre-built wheels:

.. code-block:: bash

    # Let pip find pre-built wheels
    pip install --only-binary :all: gradient-free-optimizers


macOS: SSL Certificate Errors
------------------------------

**Problem**: SSL errors when downloading from PyPI.

**Solution**: Update certificates:

.. code-block:: bash

    /Applications/Python\ 3.X/Install\ Certificates.command


Or use conda to manage SSL:

.. code-block:: bash

    conda install certifi


Linux: "No matching distribution found"
-----------------------------------------

**Problem**: pip can't find the package.

**Solution**: Update pip:

.. code-block:: bash

    pip install --upgrade pip
    pip install gradient-free-optimizers


----

Jupyter Notebook Issues
========================

Kernel Can't Find Package
--------------------------

**Problem**: Package installed but Jupyter can't import it.

**Solution**: Install in the Jupyter kernel's environment:

.. code-block:: bash

    # Find the kernel's Python
    python -m ipykernel --version

    # Install there
    python -m pip install gradient-free-optimizers

Or install directly in a notebook cell:

.. code-block:: python

    !pip install gradient-free-optimizers


----

Verification
============

After fixing installation issues, verify GFO works:

.. code-block:: python

    import gradient_free_optimizers as gfo
    import numpy as np

    print(f"GFO version: {gfo.__version__}")

    # Quick test
    from gradient_free_optimizers import RandomSearchOptimizer

    search_space = {"x": np.linspace(-1, 1, 10)}
    opt = RandomSearchOptimizer(search_space)
    opt.search(lambda p: -p["x"]**2, n_iter=10)

    print(f"Test passed! Best score: {opt.best_score}")


----

Still Having Issues?
====================

If installation problems persist:

1. **Check system compatibility**: Python 3.10+, recent pip version
2. **Try a fresh virtual environment**: Isolates dependency conflicts
3. **Report the issue**: See :ref:`troubleshooting_help`

When reporting, include:

- Python version (``python --version``)
- pip version (``pip --version``)
- Operating system
- Full error message
- Output of ``pip list``
