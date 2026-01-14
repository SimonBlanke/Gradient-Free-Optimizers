========
Examples
========

A collection of practical examples showing how to use Gradient-Free-Optimizers
for various optimization tasks.


Quick Start Examples
--------------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Simple Function Optimization
        :link: simple_optimization
        :link-type: doc

        Minimize a simple 2D function to understand the basics of GFO.

    .. grid-item-card:: ML Hyperparameter Tuning
        :link: hyperparameter_tuning
        :link-type: doc

        Tune scikit-learn model hyperparameters using various optimizers.


Advanced Examples
-----------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Using Constraints
        :link: constrained_optimization
        :link-type: doc

        Define and use parameter constraints in optimization.

    .. grid-item-card:: Ask-Tell Interface
        :link: ask_tell_example
        :link-type: doc

        Manual control over the optimization loop for custom workflows.

    .. grid-item-card:: Comparing Optimizers
        :link: comparing_optimizers
        :link-type: doc

        Benchmark different algorithms on the same problem.

    .. grid-item-card:: Memory and Warm Starts
        :link: warm_start
        :link-type: doc

        Continue optimization from previous results.


.. toctree::
    :maxdepth: 1
    :hidden:

    simple_optimization
    hyperparameter_tuning
    constrained_optimization
    ask_tell_example
    comparing_optimizers
    warm_start
