.. _about:

=====
About
=====

Gradient-Free-Optimizers is a Python library providing 22 gradient-free
optimization algorithms with a unified interface for black-box function optimization.

.. toctree::
   :maxdepth: 1

   about/team
   about/history
   about/license
   about/citation


About Gradient-Free-Optimizers
-------------------------------

Gradient-Free-Optimizers (GFO) provides a collection of optimization algorithms
designed for problems where gradients are unavailable or impractical to compute.
It serves as the optimization backend for
`Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_ and can also be
used standalone for custom optimization tasks.


Mission
^^^^^^^

GFO aims to make gradient-free optimization accessible and practical. By providing
a unified interface across 22 algorithms, it reduces the complexity of choosing
and switching between optimization methods. The library prioritizes simplicity,
transparency, and reliability.


Key Features
^^^^^^^^^^^^

- **22 Optimization Algorithms**: From simple hill climbing to advanced Bayesian
  optimization, organized into local, global, population, and sequential model-based categories.

- **Unified Interface**: All algorithms share the same API. Switch algorithms by
  changing one line of code.

- **NumPy-Based Search Spaces**: No special types or complex syntax. Define search
  spaces using familiar NumPy arrays.

- **Constraint Support**: Define constraints as simple Python functions. Invalid
  positions are automatically avoided during search.

- **Memory System**: Built-in caching prevents redundant evaluations. Critical
  for expensive objective functions like ML model training.

- **Production Ready**: Extensive test coverage across Python 3.10-3.14,
  NumPy 1.x/2.x, and all major operating systems.


Design Philosophy
^^^^^^^^^^^^^^^^^

1. **Simplicity First**: Common tasks should be straightforward. Define your
   search space, choose an optimizer, and run.

2. **Transparent Behavior**: No hidden state or implicit configuration. All
   algorithm parameters have sensible defaults but can be adjusted.

3. **Interchangeable Algorithms**: The unified interface makes it easy to
   experiment with different optimizers.

4. **Zero Dependencies**: Core library only requires NumPy. Optional features
   (progress bars, SMBO) have opt-in dependencies.


Related Projects
^^^^^^^^^^^^^^^^

GFO is part of a larger ecosystem for optimization and benchmarking:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Project
     - Description
   * - `Gradient-Free-Optimizers <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
     - Core optimization algorithms (you are here)
   * - `Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_
     - Higher-level toolkit built on GFO with parallel optimization, experiment tracking, and ML integrations
   * - `Surfaces <https://github.com/SimonBlanke/Surfaces>`_
     - Test functions and benchmark surfaces for evaluating optimization algorithms


When to Use GFO
^^^^^^^^^^^^^^^

**Use Gradient-Free-Optimizers when:**

- You need fine-grained control over the optimization process
- You're optimizing custom objective functions (not just ML models)
- You want minimal dependencies and a lightweight library
- You're building tools that need an optimization backend

**Use Hyperactive instead when:**

- You need parallel optimization across multiple CPU cores
- You want experiment tracking and result comparison
- You're primarily tuning ML models (sklearn, PyTorch, etc.)
- You need integration with other optimization frameworks (Optuna, sklearn)


Community & Support
^^^^^^^^^^^^^^^^^^^

- **GitHub**: `SimonBlanke/Gradient-Free-Optimizers <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
- **Issues**: `Report bugs or request features <https://github.com/SimonBlanke/Gradient-Free-Optimizers/issues>`_
- **Discussions**: `Ask questions <https://github.com/SimonBlanke/Gradient-Free-Optimizers/discussions>`_


How to Cite
^^^^^^^^^^^

If you use Gradient-Free-Optimizers in your research, please see :doc:`about/citation`
for citation information.
