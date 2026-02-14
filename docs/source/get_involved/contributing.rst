.. _contributing:

============
Contributing
============

Thank you for your interest in contributing to Gradient-Free-Optimizers! This guide
will help you get started with development and submit your contributions.

.. seealso::

   Also see `CONTRIBUTING.md <https://github.com/SimonBlanke/Gradient-Free-Optimizers/blob/main/CONTRIBUTING.md>`_
   in the repository for additional guidelines.

----

How to Contribute
-----------------

Contribution Workflow
^^^^^^^^^^^^^^^^^^^^^

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** with tests
5. **Run the test suite** to ensure everything works
6. **Submit a pull request** for review

Types of Contributions
^^^^^^^^^^^^^^^^^^^^^^

We welcome many types of contributions:

**Bug Fixes**
    Fix issues and improve stability. Search `existing issues <https://github.com/SimonBlanke/Gradient-Free-Optimizers/issues>`_
    for bugs to work on.

**New Features**
    Add new optimizers, features, or functionality. Discuss major features in
    an issue or discussion first.

**Documentation**
    Improve guides, examples, API docs, or fix typos. Documentation is as
    important as code!

**Tests**
    Increase test coverage or add test cases. Good tests prevent regressions.

**Performance**
    Optimize code for speed or memory. Include benchmarks showing improvements.

**Examples**
    Add examples from your field of work that incorporate GFO.

----

Development Setup
-----------------

Prerequisites
^^^^^^^^^^^^^

- Python 3.10 or higher
- Git
- pip

Setting Up Your Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Fork the repository on GitHub

2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/YOUR-USERNAME/Gradient-Free-Optimizers.git
       cd Gradient-Free-Optimizers

3. Add the upstream repository:

   .. code-block:: bash

       git remote add upstream https://github.com/SimonBlanke/Gradient-Free-Optimizers.git

4. Create a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

5. Install in development mode with dependencies:

   .. code-block:: bash

       pip install -e ".[test,docs,progress,sklearn]"

6. Verify the installation:

   .. code-block:: bash

       python -c "import gradient_free_optimizers as gfo; print(gfo.__version__)"
       pytest tests/ -v

----

Making Changes
--------------

Create a Branch
^^^^^^^^^^^^^^^

Create a branch for your work:

.. code-block:: bash

    git checkout -b fix/issue-123-memory-leak
    # or
    git checkout -b feature/add-new-optimizer

Use descriptive branch names:

- ``fix/issue-123-description`` for bug fixes
- ``feature/description`` for new features
- ``docs/description`` for documentation
- ``refactor/description`` for code refactoring


Write Code
^^^^^^^^^^

Follow these guidelines:

1. **Keep changes focused**: One PR should address one issue/feature
2. **Follow existing code style**: Match the surrounding code
3. **Add docstrings**: Use NumPy-style docstrings for new functions/classes
4. **Add type hints**: Use Python type annotations when possible
5. **Handle edge cases**: Consider boundary conditions and errors

Example of good code structure:

.. code-block:: python

    def new_function(param1: np.ndarray, param2: int) -> float:
        """Brief description of what this function does.

        More detailed explanation if needed.

        Parameters
        ----------
        param1 : np.ndarray
            Description of param1.
        param2 : int
            Description of param2.

        Returns
        -------
        float
            Description of return value.

        Examples
        --------
        >>> result = new_function(np.array([1, 2, 3]), 5)
        >>> print(result)
        10.5
        """
        # Implementation here
        return result


Add Tests
^^^^^^^^^

All new code should have tests:

.. code-block:: python

    # tests/test_new_feature.py
    import numpy as np
    from gradient_free_optimizers import NewOptimizer


    def test_new_optimizer_basic():
        """Test basic functionality of NewOptimizer."""
        search_space = {"x": np.linspace(-5, 5, 20)}

        def objective(params):
            return -params["x"]**2

        opt = NewOptimizer(search_space)
        opt.search(objective, n_iter=10)

        assert opt.best_score < 0
        assert "x" in opt.best_para


    def test_new_optimizer_with_constraints():
        """Test NewOptimizer respects constraints."""
        # Test implementation


Run Tests
^^^^^^^^^

Run tests locally before submitting:

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=gradient_free_optimizers --cov-report=html

    # Run specific test file
    pytest tests/test_new_feature.py

    # Run tests matching a pattern
    pytest -k "test_new_optimizer"

    # Run with verbose output
    pytest -v


Update Documentation
^^^^^^^^^^^^^^^^^^^^

If you're adding a new feature:

1. **Add docstrings** to all public functions/classes
2. **Add examples** in the docstrings
3. **Update relevant docs** in ``docs/source/``
4. **Add to API reference** if it's a new optimizer

Build docs locally to verify:

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    make html
    # Open docs/build/html/index.html in a browser

----

Submitting Changes
------------------

Commit Your Changes
^^^^^^^^^^^^^^^^^^^

Write clear commit messages:

.. code-block:: bash

    git add .
    git commit -m "[Fix] Resolve memory leak in search data collection

    - Clear search_data DataFrame periodically
    - Add test for long-running optimizations
    - Closes #123"

Commit message guidelines:

- Use tags: ``[Fix]``, ``[Feature]``, ``[Refactor]``, ``[Docs]``, ``[Test]``
- First line: Brief summary (< 72 characters)
- Blank line, then detailed explanation if needed
- Reference issues with ``Closes #123`` or ``Fixes #123``


Push Your Branch
^^^^^^^^^^^^^^^^

.. code-block:: bash

    git push origin fix/issue-123-memory-leak


Create a Pull Request
^^^^^^^^^^^^^^^^^^^^^^

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill in the PR template:

   - **Title**: Use tag prefix (``[Fix]``, ``[Feature]``, etc.)
   - **Description**: Explain what and why, not just what
   - **Link issues**: Reference related issues
   - **Testing**: Describe how you tested the changes

Example PR description:

.. code-block:: markdown

    ## Description
    Fixes memory leak in search data collection for long-running optimizations.

    ## Changes
    - Add periodic clearing of search_data DataFrame
    - Implement configurable memory threshold
    - Add tests for long-running scenarios

    ## Motivation
    Users reported memory errors after 10,000+ iterations.

    ## Testing
    - Added `test_long_running_optimization` that verifies memory stays bounded
    - Ran 50,000 iteration test with memory profiling
    - All existing tests pass

    Closes #123


PR Review Process
^^^^^^^^^^^^^^^^^

1. **Automated checks** run (tests, linting)
2. **Maintainer reviews** code and provides feedback
3. **You address** any requested changes
4. **Maintainer approves** and merges

Tips for faster reviews:

- Keep PRs focused and reasonably sized
- Respond promptly to feedback
- Be open to suggestions
- Update your branch if requested

----

Code Style
----------

Formatting
^^^^^^^^^^

GFO follows PEP 8 with some flexibility. Key points:

- 4 spaces for indentation (not tabs)
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names
- Add blank lines to separate logical sections

Naming Conventions
^^^^^^^^^^^^^^^^^^

- **Classes**: ``PascalCase`` (e.g., ``HillClimbingOptimizer``)
- **Functions/methods**: ``snake_case`` (e.g., ``search``)
- **Constants**: ``UPPER_SNAKE_CASE`` (e.g., ``DEFAULT_N_ITER``)
- **Private members**: ``_leading_underscore`` (e.g., ``_internal_method``)


Imports
^^^^^^^

Organize imports in this order:

.. code-block:: python

    # Standard library
    import os
    import sys
    from typing import Dict, List

    # Third-party
    import numpy as np
    import pandas as pd

    # Local
    from gradient_free_optimizers.base import BaseOptimizer
    from gradient_free_optimizers.utils import validate_search_space


----

Testing Guidelines
------------------

What to Test
^^^^^^^^^^^^

- **Happy path**: Normal usage scenarios
- **Edge cases**: Boundary conditions, empty inputs
- **Error cases**: Invalid inputs should raise appropriate errors
- **Integration**: Features work together correctly

Test Structure
^^^^^^^^^^^^^^

.. code-block:: python

    def test_feature_name():
        """Test description of what is being tested."""
        # Arrange: Set up test data
        search_space = {"x": np.array([1, 2, 3])}

        # Act: Execute the code being tested
        opt = Optimizer(search_space)
        result = opt.some_method()

        # Assert: Verify the results
        assert result == expected_value


Running Specific Tests
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Single test
    pytest tests/test_file.py::test_specific_function

    # Test class
    pytest tests/test_file.py::TestClassName

    # With debugging output
    pytest tests/test_file.py -v -s


----

Adding a New Optimizer
----------------------

If you're contributing a new optimizer:

1. **Inherit from base class**:

   .. code-block:: python

       from gradient_free_optimizers.optimizers.base_optimizer import BaseOptimizer

       class MyOptimizer(BaseOptimizer):
           def __init__(self, search_space, **kwargs):
               super().__init__(search_space, **kwargs)
               # Your initialization

2. **Implement required methods**:

   - ``_init_position()`` - Initialize starting position
   - ``_iterate()`` - Single optimization step

3. **Add to exports**:

   In ``src/gradient_free_optimizers/__init__.py``:

   .. code-block:: python

       from .optimizers.my_optimizer import MyOptimizer

       __all__ = [
           # ... existing optimizers
           "MyOptimizer",
       ]

4. **Write tests** in ``tests/test_my_optimizer.py``

5. **Add documentation** in ``docs/source/user_guide/optimizers/category/my_optimizer.rst``

----

Getting Help
------------

If you need help with your contribution:

1. **Ask in the PR**: Comment on your pull request
2. **Open a discussion**: Use GitHub Discussions for questions
3. **Join the community**: See :ref:`troubleshooting_help`
4. **Email maintainer**: simon.blanke@yahoo.com for private matters

----

Code of Conduct
---------------

All contributors must follow our :doc:`code_of_conduct`. Be respectful,
welcoming, and constructive in all interactions.

----

License
-------

By contributing, you agree that your contributions will be licensed under the
MIT License, the same as the rest of the project.

----

Thank You!
----------

Thank you for contributing to Gradient-Free-Optimizers! Your efforts help make
optimization more accessible to everyone.
