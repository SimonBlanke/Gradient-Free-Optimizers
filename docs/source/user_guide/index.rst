==========
User Guide
==========

This guide covers all the features and concepts you need to effectively use
Gradient-Free-Optimizers. Whether you're doing simple function optimization
or complex hyperparameter tuning, this guide will help you get the most out
of the library.


Core Concepts
-------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: Search Spaces
        :link: search_spaces
        :link-type: doc

        How to define parameter spaces using NumPy arrays. Covers continuous,
        discrete, and categorical parameters.

    .. grid-item-card:: Objective Functions
        :link: objective_functions
        :link-type: doc

        Writing objective functions, handling return values, and working with
        expensive function evaluations.

    .. grid-item-card:: Optimizer Selection
        :link: optimizer_selection
        :link-type: doc

        Choosing the right algorithm for your problem. When to use local vs.
        global, single vs. population-based methods.


Features
--------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Constraints
        :link: constraints
        :link-type: doc

        Define parameter constraints using Python functions. The optimizer
        automatically respects constraints during search.

    .. grid-item-card:: Initialization
        :link: initialization
        :link-type: doc

        Control how the search starts: grid points, random sampling, vertices,
        or custom warm-start positions.

    .. grid-item-card:: Memory & Caching
        :link: memory
        :link-type: doc

        Automatic caching of function evaluations and warm-starting from
        previous optimization runs.

    .. grid-item-card:: Stopping Conditions
        :link: stopping_conditions
        :link-type: doc

        Stop optimization based on iterations, time, target score, or early
        stopping when progress stalls.


Interfaces
----------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: The search() Method
        :link: search_interface
        :link-type: doc

        The simple, all-in-one interface for running optimization. Best for
        most use cases.

    .. grid-item-card:: Ask-Tell Interface
        :link: ask_tell
        :link-type: doc

        Manual control over the optimization loop. Useful for distributed
        computing, custom logging, or integration with external systems.


Algorithms
----------

Detailed documentation for all 22 optimization algorithms:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Local Search Algorithms
        :link: optimizers/local/index
        :link-type: doc
        :class-card: sd-border-start sd-border-danger

        **5 algorithms** including Hill Climbing, Simulated Annealing,
        and Downhill Simplex.

    .. grid-item-card:: Global Search Algorithms
        :link: optimizers/global/index
        :link-type: doc
        :class-card: sd-border-start sd-border-success

        **8 algorithms** including Random Search, Grid Search,
        Pattern Search, and DIRECT.

    .. grid-item-card:: Population-Based Algorithms
        :link: optimizers/population/index
        :link-type: doc
        :class-card: sd-border-start sd-border-primary

        **6 algorithms** including Particle Swarm, Genetic Algorithm,
        and Differential Evolution.

    .. grid-item-card:: Sequential Model-Based
        :link: optimizers/smbo/index
        :link-type: doc
        :class-card: sd-border-start sd-border-warning

        **4 algorithms** including Bayesian Optimization, TPE,
        and Forest Optimizer.


.. toctree::
    :maxdepth: 2
    :hidden:

    search_spaces
    objective_functions
    optimizer_selection
    constraints
    initialization
    memory
    stopping_conditions
    search_interface
    ask_tell
    optimizers/index
