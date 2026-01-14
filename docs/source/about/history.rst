.. _about_history:

=======
History
=======

The story of Gradient-Free-Optimizers and its evolution from 2019 to present.

----

Origins (2019)
--------------

Gradient-Free-Optimizers was created in 2019 by Simon Blanke as the optimization
backend for Hyperactive. The goal was to provide a unified interface to multiple
gradient-free optimization algorithms, making it easy to experiment with different
approaches without changing code.

**Initial algorithms**:

- Hill Climbing
- Random Search
- Grid Search
- Simulated Annealing
- Particle Swarm Optimization

----

Growth (2020-2021)
------------------

The library expanded to include more sophisticated algorithms:

**Sequential Model-Based Optimization**:

- Bayesian Optimization with Gaussian Processes
- Tree-structured Parzen Estimator (TPE)
- Random Forest Optimizer

**Population Methods**:

- Genetic Algorithm
- Evolution Strategy
- Differential Evolution

**Global Methods**:

- DIRECT algorithm
- Lipschitz Optimization
- Pattern Search

----

Maturation (2022-2023)
----------------------

Focus shifted to stability, testing, and user experience:

- Comprehensive test suite across Python versions
- Improved documentation
- Performance optimizations
- Better error handling
- NumPy 2.0 compatibility
- Constraint support enhancements

----

Recent Developments (2024-2025)
-------------------------------

Continued refinement and modernization:

- Python 3.10+ support
- Enhanced type annotations
- Improved memory efficiency
- Better API consistency
- Expanded documentation with examples
- Community contributions

----

Design Philosophy Evolution
---------------------------

**Core principles maintained**:

1. **Simplicity**: NumPy arrays for search spaces, no complex syntax
2. **Consistency**: All algorithms share the same interface
3. **Transparency**: No hidden state or magic
4. **Minimal dependencies**: Core library only needs NumPy

**Improvements over time**:

- Better separation of concerns
- More intuitive parameter names
- Clearer error messages
- Better documentation

----

Impact
------

**Usage**:

- Used as backend for Hyperactive
- Applied in research and industry
- Taught in optimization courses
- Featured in academic papers

**Community**:

- Active GitHub repository
- Regular issue discussion and resolution
- Contributions from researchers and practitioners
- Integration with other tools

----

Looking Forward
---------------

**Future directions**:

- Additional optimization algorithms
- Performance improvements
- Enhanced constraint handling
- Better parallel optimization support (via Hyperactive)
- Expanded documentation and examples

----

Timeline
--------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Year
     - Milestone
   * - 2019
     - Initial release with 5 algorithms
   * - 2020
     - Added SMBO algorithms (Bayesian, TPE)
   * - 2021
     - Reached 20+ algorithms
   * - 2022
     - Comprehensive testing infrastructure
   * - 2023
     - NumPy 2.0 compatibility
   * - 2024
     - Python 3.10+ focus, improved docs
   * - 2025
     - Continued refinement and community growth

----

Releases
--------

For detailed release notes and version history, see:

- :doc:`changelog`
- `GitHub Releases <https://github.com/SimonBlanke/Gradient-Free-Optimizers/releases>`_

----

Thank You
---------

Thank you to everyone who has used, tested, reported issues, contributed code,
or spread the word about Gradient-Free-Optimizers over the years!
