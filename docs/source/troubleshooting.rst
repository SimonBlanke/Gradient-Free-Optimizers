.. _troubleshooting_guide:

===============
Troubleshooting
===============

This guide helps you diagnose and fix common issues with Gradient-Free-Optimizers.
If you don't find a solution here, see :ref:`troubleshooting_help` for support options.

.. toctree::
   :maxdepth: 1

   troubleshooting/installation
   troubleshooting/runtime_errors
   troubleshooting/performance
   troubleshooting/results
   troubleshooting/getting_help


Overview
--------

:ref:`troubleshooting_installation`
    Import errors, missing modules, dependency conflicts, and version compatibility.

:ref:`troubleshooting_runtime`
    AttributeError, TypeError, ValueError, and other runtime errors during optimization.

:ref:`troubleshooting_performance`
    Slow optimization, high memory usage, and performance bottlenecks.

:ref:`troubleshooting_results`
    Unexpected results, local optima, reproducibility issues, and score problems.

:ref:`troubleshooting_help`
    Where to get additional support and how to report issues.

----

Quick Diagnosis
---------------

.. grid:: 2 2 2 2
   :gutter: 3

   .. grid-item-card:: Can't import GFO

      Check Python version (needs 3.10+),
      verify installation with ``pip list | grep gradient``

      +++
      :ref:`troubleshooting_installation`

   .. grid-item-card:: Getting runtime errors

      Check search space definition,
      objective function signature, and parameter types

      +++
      :ref:`troubleshooting_runtime`

   .. grid-item-card:: Optimization is slow

      Profile objective function,
      reduce search space size, or choose faster algorithm

      +++
      :ref:`troubleshooting_performance`

   .. grid-item-card:: Results look wrong

      Verify maximization vs. minimization,
      check search space bounds, increase iterations

      +++
      :ref:`troubleshooting_results`
