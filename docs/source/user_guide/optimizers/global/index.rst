=========================
Global Search Algorithms
=========================

Global search algorithms are designed to explore the entire search space rather
than focusing on a single neighborhood. They are better at avoiding local optima
but may be slower to converge to the exact optimum.


Overview
--------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Algorithm
      - Description
    * - :doc:`random_search`
      - Pure random sampling. Simple but effective baseline.
    * - :doc:`grid_search`
      - Systematic traversal of the search space.
    * - :doc:`random_restart`
      - Hill climbing with periodic random restarts.
    * - :doc:`random_annealing`
      - Large initial steps that gradually decrease.
    * - :doc:`pattern_search`
      - Structured exploration using geometric patterns.
    * - :doc:`powells_method`
      - Sequential optimization along each dimension.
    * - :doc:`lipschitz`
      - Uses Lipschitz continuity bounds to prune search.
    * - :doc:`direct`
      - Divides space into regions, exploring the most promising.


When to Use Global Search
-------------------------

**Good for:**

- Unknown landscapes without prior information
- Problems with many local optima
- Establishing a baseline for comparison
- Cases where you need guaranteed coverage

**Not ideal for:**

- Very expensive objective functions (use SMBO instead)
- Fine-tuning when you have a good starting point
- Very high-dimensional spaces


Algorithm Comparison
--------------------

.. list-table::
    :header-rows: 1
    :widths: 20 20 20 40

    * - Algorithm
      - Deterministic?
      - Coverage
      - Best Use Case
    * - Random Search
      - No
      - Probabilistic
      - Baseline, high dimensions
    * - Grid Search
      - Yes
      - Complete
      - Small, discrete spaces
    * - Random Restart
      - No
      - Hybrid
      - Multi-modal with local structure
    * - Random Annealing
      - No
      - Adaptive
      - Broad to focused search
    * - Pattern Search
      - Yes
      - Structured
      - Derivative-free optimization
    * - Powell's Method
      - Yes
      - Dimensional
      - Separable functions
    * - Lipschitz
      - Yes
      - Bounded
      - Functions with known smoothness
    * - DIRECT
      - Yes
      - Hierarchical
      - Guaranteed convergence


Conceptual Comparison
---------------------

.. figure:: /_static/diagrams/global_search_comparison.svg
    :alt: Global search algorithm comparison
    :align: center

    How each global search algorithm covers a 2D search space. Random Search
    samples uniformly, Grid Search covers systematically, Pattern Search uses
    geometric patterns, and DIRECT divides hierarchically.


Visualization
-------------

.. grid:: 2
    :gutter: 3

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/random_search_sphere_function_.gif
            :alt: Random Search on Sphere function

            Random Search explores the space uniformly without bias.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/grid_search_sphere_function_.gif
            :alt: Grid Search on Sphere function

            Grid Search systematically covers the space.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/pattern_search_sphere_function_.gif
            :alt: Pattern Search on Sphere function

            Pattern Search uses structured geometric patterns.

    .. grid-item::
        :columns: 6

        .. figure:: /_static/gifs/direct_algorithm_sphere_function_.gif
            :alt: DIRECT on Sphere function

            DIRECT divides space hierarchically, focusing on promising regions.


Algorithms
----------

.. toctree::
    :maxdepth: 1

    random_search
    grid_search
    random_restart
    random_annealing
    pattern_search
    powells_method
    lipschitz
    direct
