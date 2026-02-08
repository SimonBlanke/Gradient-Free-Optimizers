:html_theme.sidebar_secondary.remove:

===========
Get Started
===========

Gradient-Free-Optimizers provides 22 optimization algorithms behind a simple,
Pythonic API. These pages cover everything you need to go from installation to
running your first optimization.


.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Introduction
        :link: introduction
        :link-type: doc

        Install the library, run your first optimization, and learn the
        core workflow: search space, objective function, optimizer, results.

    .. grid-item-card:: Optimization Algorithms
        :link: algorithms
        :link-type: doc

        Overview of all 22 algorithms organized by category: local, global,
        population-based, and sequential model-based.

    .. grid-item-card:: Mixed Search Spaces
        :link: search_spaces
        :link-type: doc

        Continuous, discrete, and categorical dimensions freely mixed in a
        single search space dictionary.

    .. grid-item-card:: Minimal Dependencies
        :link: dependencies
        :link-type: doc

        Only NumPy and pandas required. No heavy frameworks,
        no compiled extensions, fast installation everywhere.

    .. grid-item-card:: Pythonic API
        :link: pythonic_api
        :link-type: doc

        Dict search spaces, callable objectives, DataFrame results.
        Standard Python data structures, no special objects.


.. toctree::
    :maxdepth: 2
    :hidden:

    introduction
    algorithms
    search_spaces
    dependencies
    pythonic_api
