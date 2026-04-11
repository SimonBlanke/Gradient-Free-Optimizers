.. _user_guide:

==========
User Guide
==========

Master gradient-free optimization with GFO. This guide covers core concepts,
algorithm selection, and advanced features for efficient black-box optimization.

.. tip::

   New to GFO? Follow this path through the guide:

   1. :doc:`user_guide/search_spaces` -- learn how to define parameter ranges
   2. :doc:`user_guide/objective_functions` -- write functions to optimize
   3. :doc:`user_guide/optimizer_selection` -- choose the right algorithm for your problem

----

How GFO Works
=============

Gradient-Free-Optimizers provides a simple pattern: define your search space,
choose an optimizer, and run. All algorithms share the same interface, making
it easy to experiment with different approaches.

.. code-block:: python

    # The GFO optimization pattern
    from gradient_free_optimizers import BayesianOptimizer
    import numpy as np

    # 1. Define where to search
    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
    }

    # 2. Choose how to optimize
    opt = BayesianOptimizer(search_space)

    # 3. Run optimization
    opt.search(objective_function, n_iter=100)

    # 4. Get results
    print(opt.best_para)   # Best parameters
    print(opt.best_score)  # Best score achieved

----

Core Concepts
=============

Every optimization in GFO involves three components:

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card:: Search Spaces
      :class-card: sd-border-primary
      :link: user_guide/search_spaces
      :link-type: doc

      **Where to search**

      Parameter ranges defined as NumPy arrays. Supports continuous,
      discrete, and categorical parameters.

   .. grid-item-card:: Optimizers
      :class-card: sd-border-success
      :link: user_guide/optimizers/index
      :link-type: doc

      **How to optimize**

      The algorithm that explores your search space. Choose from 22 algorithms
      across local search, global search, population methods, and SMBO.

   .. grid-item-card:: Objective Functions
      :class-card: sd-border-warning
      :link: user_guide/objective_functions
      :link-type: doc

      **What to optimize**

      Your function that takes parameters and returns a score to maximize.

----

Guide Sections
==============

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Search Spaces
      :link: user_guide/search_spaces
      :link-type: doc

      Define parameter ranges with NumPy arrays.
      **Start here** for the fundamentals.

   .. grid-item-card:: Objective Functions
      :link: user_guide/objective_functions
      :link-type: doc

      Writing functions to optimize and handling
      expensive evaluations.

   .. grid-item-card:: Optimizer Selection
      :link: user_guide/optimizer_selection
      :link-type: doc

      Choosing the right algorithm for your
      problem type.

   .. grid-item-card:: Constraints
      :link: user_guide/constraints
      :link-type: doc

      Restrict the search space with
      constraint functions.

   .. grid-item-card:: Initialization
      :link: user_guide/initialization
      :link-type: doc

      Control how the search starts:
      grid, random, or warm-start.

   .. grid-item-card:: Memory & Caching
      :link: user_guide/memory
      :link-type: doc

      Cache evaluations and warm-start
      from previous runs.

   .. grid-item-card:: Stopping Conditions
      :link: user_guide/stopping_conditions
      :link-type: doc

      Stop based on iterations, time,
      or early stopping.

   .. grid-item-card:: Search Interface
      :link: user_guide/search_interface
      :link-type: doc

      Using the simple ``.search()``
      method.

----

Algorithms
==========

GFO provides 22 optimization algorithms organized into four categories:

.. grid:: 2 2 2 2
   :gutter: 3

   .. grid-item-card:: Local Search
      :link: user_guide/optimizers/local/index
      :link-type: doc
      :class-card: sd-border-start sd-border-danger

      **5 algorithms** for exploiting promising regions

      Hill Climbing variants, Simulated Annealing,
      and Downhill Simplex.

   .. grid-item-card:: Global Search
      :link: user_guide/optimizers/global/index
      :link-type: doc
      :class-card: sd-border-start sd-border-success

      **8 algorithms** for broad exploration

      Random Search, Grid Search, Pattern Search,
      DIRECT, and more.

   .. grid-item-card:: Population-Based
      :link: user_guide/optimizers/population/index
      :link-type: doc
      :class-card: sd-border-start sd-border-primary

      **6 algorithms** using collective intelligence

      Particle Swarm, Genetic Algorithm, Evolution
      Strategy, Differential Evolution.

   .. grid-item-card:: Sequential Model-Based
      :link: user_guide/optimizers/smbo/index
      :link-type: doc
      :class-card: sd-border-start sd-border-warning

      **3 algorithms** that learn from evaluations

      Bayesian Optimization, TPE, and Forest Optimizer.

----

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/search_spaces
   user_guide/objective_functions
   user_guide/optimizer_selection
   user_guide/constraints
   user_guide/initialization
   user_guide/memory
   user_guide/stopping_conditions
   user_guide/search_interface

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Optimization Algorithms

   Overview <user_guide/optimizers/index>
   user_guide/optimizers/local/index
   user_guide/optimizers/global/index
   user_guide/optimizers/population/index
   user_guide/optimizers/smbo/index
