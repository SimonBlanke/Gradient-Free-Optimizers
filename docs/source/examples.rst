.. _examples:

========
Examples
========

Practical examples showing how to use Gradient-Free-Optimizers for various
optimization tasks. All examples are complete and ready to run.

----

Getting Started
===============

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Simple Function Optimization
      :link: examples/simple_optimization
      :link-type: doc

      **Start here**: Minimize a simple 2D function to learn the basics
      of GFO. Covers search space definition, running optimization,
      and accessing results.

   .. grid-item-card:: ML Hyperparameter Tuning
      :link: examples/hyperparameter_tuning
      :link-type: doc

      Tune scikit-learn model hyperparameters using Bayesian Optimization.
      Shows how to define search spaces for real ML models and interpret
      cross-validation scores.

----

Advanced Features
=================

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Constrained Optimization
      :link: examples/constrained_optimization
      :link-type: doc

      Define parameter constraints to restrict the search space. Perfect
      for problems with physical constraints or valid parameter combinations.

   .. grid-item-card:: Ask-Tell Interface
      :link: examples/ask_tell_example
      :link-type: doc

      Manual control over the optimization loop. Useful for distributed
      computing, custom logging, or integration with external systems.

   .. grid-item-card:: Comparing Algorithms
      :link: examples/comparing_optimizers
      :link-type: doc

      Benchmark different optimizers on the same problem to find the best
      algorithm for your use case. Includes visualization of results.

   .. grid-item-card:: Memory and Warm Starts
      :link: examples/warm_start
      :link-type: doc

      Cache expensive evaluations and continue optimization from previous
      runs. Essential for long-running optimizations.

----

By Use Case
===========

.. tab-set::

   .. tab-item:: Quick Prototyping

      Use :doc:`Random Search <examples/simple_optimization>` for fast exploration
      or :doc:`Hill Climbing <examples/simple_optimization>` for smooth functions.

   .. tab-item:: Hyperparameter Tuning

      See the :doc:`ML example <examples/hyperparameter_tuning>` for optimizing
      scikit-learn models with Bayesian Optimization.

   .. tab-item:: Constrained Problems

      Learn how to handle constraints in :doc:`this example <examples/constrained_optimization>`.

   .. tab-item:: Expensive Functions

      Use the :doc:`warm start example <examples/warm_start>` to cache results
      and avoid redundant evaluations.

----

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/index
