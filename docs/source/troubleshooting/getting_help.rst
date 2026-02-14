.. _troubleshooting_help:

============
Getting Help
============

Where to find support and how to report issues with Gradient-Free-Optimizers.

----

Before Asking for Help
=======================

Check These Resources First
----------------------------

1. **Documentation**

   - :doc:`../user_guide/index` - Core concepts and features
   - :doc:`../examples/index` - Working code examples
   - :doc:`../faq/index` - Common questions
   - :doc:`../troubleshooting` - Other troubleshooting pages

2. **Existing Issues**

   Search `GitHub Issues <https://github.com/SimonBlanke/Gradient-Free-Optimizers/issues>`_
   to see if your problem has been reported.

3. **Search Data**

   Inspect ``opt.search_data`` to understand what the optimizer is doing:

   .. code-block:: python

       opt.search(objective, n_iter=100)
       print(opt.search_data.head(20))  # First 20 evaluations
       print(opt.search_data.describe())  # Statistical summary


Create a Minimal Example
-------------------------

Before reporting an issue, create the smallest code that reproduces the problem:

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    # Minimal objective function
    def objective(params):
        return -(params["x"]**2)

    # Minimal search space
    search_space = {"x": np.linspace(-5, 5, 20)}

    # The issue
    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=10)  # Fails here

Remove unnecessary code, custom data, and complex ML models. Use simple
synthetic functions when possible.

----

Getting Help
============

GitHub Issues
-------------

**Best for**: Bug reports, feature requests, installation problems

1. Go to `GitHub Issues <https://github.com/SimonBlanke/Gradient-Free-Optimizers/issues>`_
2. Click "New Issue"
3. Choose the appropriate template:

   - **Bug Report**: Something isn't working
   - **Feature Request**: Suggest an enhancement
   - **Question**: Ask about usage

4. Fill in the template with details

**What to include**:

.. code-block:: markdown

    ### Environment
    - GFO version: (run `gradient_free_optimizers.__version__`)
    - Python version: (run `python --version`)
    - NumPy version: (run `import numpy; print(numpy.__version__)`)
    - Operating System: (e.g., Ubuntu 22.04, Windows 11, macOS 13)

    ### Description
    Clear description of the problem or request.

    ### Minimal Code Example
    ```python
    # Smallest code that shows the problem
    import gradient_free_optimizers
    # ...
    ```

    ### Expected Behavior
    What you expected to happen.

    ### Actual Behavior
    What actually happened (include full error traceback).

    ### Additional Context
    Any other relevant information.


GitHub Discussions
------------------

**Best for**: General questions, design decisions, ideas

1. Go to `GitHub Discussions <https://github.com/SimonBlanke/Gradient-Free-Optimizers/discussions>`_
2. Click "New Discussion"
3. Choose a category:

   - **Q&A**: Ask how to do something
   - **Ideas**: Suggest improvements
   - **Show and Tell**: Share your GFO projects
   - **General**: Other discussions


Email
-----

**For**: Private issues, security concerns, sensitive topics

- Email: simon.blanke@yahoo.com
- Include "GFO" in the subject line

----

Reporting Bugs
==============

Good Bug Report Checklist
--------------------------

A good bug report includes:

✓ **GFO version and environment** (Python, NumPy, OS)

✓ **Minimal code example** (< 20 lines if possible)

✓ **Full error traceback** (the complete error message)

✓ **Expected vs. actual behavior** (what should happen vs. what does happen)

✓ **Steps to reproduce** (how to trigger the bug)

Example Bug Report
------------------

**Good** (all info included):

.. code-block:: markdown

    ## Bug: KeyError when using constraints

    **Environment:**
    - GFO: 1.5.0
    - Python: 3.11.2
    - NumPy: 1.24.3
    - OS: Ubuntu 22.04

    **Minimal Example:**
    ```python
    import numpy as np
    from gradient_free_optimizers import HillClimbingOptimizer

    def objective(params):
        return -params["x"]**2

    def constraint(params):
        return params["x"] > 0

    search_space = {"x": np.linspace(-5, 5, 20)}

    opt = HillClimbingOptimizer(search_space, constraints=[constraint])
    opt.search(objective, n_iter=10)
    ```

    **Error:**
    ```
    KeyError: 'x'
    Traceback (most recent call last):
      File "test.py", line 12, in <module>
        opt.search(objective, n_iter=10)
      File ".../search.py", line 45, in search
        valid = constraint(params)
      File "test.py", line 6, in constraint
        return params["x"] > 0
    KeyError: 'x'
    ```

    **Expected:** Should optimize respecting the constraint.

    **Actual:** KeyError when evaluating constraint.

**Bad** (missing info):

.. code-block:: markdown

    ## It doesn't work

    I tried using constraints but got an error. How do I fix this?

----

Feature Requests
================

Suggesting New Features
-----------------------

When requesting a feature:

1. **Check if it exists**: Review documentation and existing issues
2. **Explain the use case**: Why is this feature needed?
3. **Describe the API**: How should it work?
4. **Consider alternatives**: Have you tried workarounds?

Example Feature Request:

.. code-block:: markdown

    ## Feature: Support for conditional parameters

    **Use Case:**
    When optimizing neural networks, some parameters only apply to
    certain architectures. For example, `dropout_rate` only matters
    when `use_dropout=True`.

    **Proposed API:**
    ```python
    search_space = {
        "use_dropout": np.array([True, False]),
        "dropout_rate": {
            "values": np.linspace(0, 0.5, 20),
            "condition": lambda p: p["use_dropout"] == True
        }
    }
    ```

    **Alternatives Tried:**
    Currently using constraints, but this is hacky:
    ```python
    def constraint(params):
        if not params["use_dropout"]:
            return params["dropout_rate"] == 0.0
        return True
    ```

    **Benefit:**
    Cleaner API and potentially more efficient optimization by
    not evaluating irrelevant parameters.

----

Contributing
============

Want to fix a bug or add a feature yourself?

1. **Read the Contributing Guide**:
   `CONTRIBUTING.md <https://github.com/SimonBlanke/Gradient-Free-Optimizers/blob/master/CONTRIBUTING.md>`_

2. **Set up development environment**:

   .. code-block:: bash

       git clone https://github.com/SimonBlanke/Gradient-Free-Optimizers.git
       cd Gradient-Free-Optimizers
       pip install -e ".[test,docs]"
       pytest tests/  # Run tests

3. **Make your changes** in a new branch

4. **Add tests** for new features or bug fixes

5. **Submit a Pull Request** with a clear description

See :doc:`../get_involved/contributing` for detailed guidelines.

----

Community Guidelines
====================

When asking for help or reporting issues:

**Do:**

- Be respectful and patient
- Provide complete information
- Follow up if you find a solution
- Help others when you can

**Don't:**

- Demand immediate responses
- Post duplicate issues
- Share private/proprietary code without permission
- Post links to external sites without context

See :doc:`../get_involved/code_of_conduct` for full community guidelines.

----

Response Time
=============

- **Issues**: Usually within 1-3 days
- **Pull Requests**: Usually within 1 week
- **Discussions**: Varies, community-driven

GFO is maintained primarily by one person, so please be patient.

----

Where is the documentation for older versions?
===============================================

Documentation for previous versions of GFO is available at:
`Legacy Documentation (v1.x) <https://simonblanke.github.io/gradient-free-optimizers-documentation/1.5/>`_

----

Related Resources
=================

**Documentation**

- GitHub: https://github.com/SimonBlanke/Gradient-Free-Optimizers

**Related Projects**

- `Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_ - Higher-level optimization toolkit
- `Surfaces <https://github.com/SimonBlanke/Surfaces>`_ - Test functions for benchmarking

**External Resources**

- Stack Overflow: Use tag ``gradient-free-optimizers``
- Reddit: r/MachineLearning for ML-related questions

----

Commercial Support
==================

For commercial support, consulting, or custom development:

- Email: simon.blanke@yahoo.com
- Include "Commercial Support" in subject line
- Describe your needs and timeline

Options include:

- Custom algorithm development
- Performance optimization
- Integration with your systems
- Training and workshops
- Priority bug fixes

----

Thank You!
==========

Thank you for using Gradient-Free-Optimizers! Your feedback and contributions
help make the library better for everyone.

.. note::

   The maintainers and contributors work on GFO in their spare time.
   We appreciate your patience and understanding!
