==========================
Frequently Asked Questions
==========================


General
-------

**What makes GFO different from other optimization libraries?**

GFO focuses on simplicity and transparency:

- Search spaces are just NumPy arrays (no special types)
- All 22 algorithms share the same interface
- Pure Python with minimal dependencies
- Serves as the backend for Hyperactive


**When should I use GFO vs. Hyperactive?**

- Use **GFO** when you need fine-grained control over optimization
- Use **Hyperactive** for higher-level features like parallel optimization,
  experiment tracking, and ML framework integrations


**Does GFO support parallel optimization?**

GFO itself runs single-threaded. For parallel optimization, use Hyperactive
which builds on GFO and adds multi-process support.


Algorithms
----------

**Which optimizer should I start with?**

- For quick exploration: ``RandomSearchOptimizer``
- For smooth functions: ``HillClimbingOptimizer``
- For expensive evaluations: ``BayesianOptimizer``
- When unsure: ``SimulatedAnnealingOptimizer`` (good general-purpose)


**My optimizer gets stuck in local optima. What should I do?**

Try these approaches:

1. Use ``SimulatedAnnealingOptimizer`` with high ``start_temp``
2. Use population-based methods like ``ParticleSwarmOptimizer``
3. Increase ``rand_rest_p`` to add random restarts
4. Use ``RandomRestartHillClimbingOptimizer``


**How many iterations do I need?**

It depends on:

- Search space size (more dimensions = more iterations)
- Function complexity (more local optima = more iterations)
- Algorithm choice (SMBO needs fewer iterations for expensive functions)

Rule of thumb: Start with 100-500 iterations and increase if needed.


Search Spaces
-------------

**How do I define a log-scale search space?**

Use ``np.logspace``:

.. code-block:: python

    search_space = {
        "learning_rate": np.logspace(-4, -1, 30),  # 0.0001 to 0.1
    }


**Can I use categorical parameters?**

Yes, use a NumPy array of strings:

.. code-block:: python

    search_space = {
        "optimizer": np.array(["adam", "sgd", "rmsprop"]),
    }


**How do I handle conditional parameters?**

GFO doesn't have native conditional support. Options:

1. Include all parameters and handle unused ones in your objective
2. Use constraints to enforce valid combinations


Performance
-----------

**Why is my optimization slow?**

Check these factors:

1. **Objective function**: The optimizer is rarely the bottleneck
2. **Memory**: Disable with ``memory=False`` if not needed
3. **SMBO overhead**: GP training is O(n^3); use Forest for many iterations
4. **Search space size**: Coarser grids are faster


**How can I speed up SMBO optimizers?**

- Use ``ForestOptimizer`` instead of ``BayesianOptimizer`` for 100+ iterations
- Reduce candidate sampling for Lipschitz/DIRECT
- Consider using simpler algorithms for cheap objective functions


Troubleshooting
---------------

**I get NaN or Inf scores. What's happening?**

GFO handles invalid scores gracefully. Check:

1. Your objective function for numerical issues
2. Parameter combinations that cause errors
3. Consider using constraints to avoid invalid regions


**The optimizer doesn't find the optimal value I expect.**

Possible causes:

1. Not enough iterations
2. Search space doesn't include the optimum
3. Using the wrong optimizer for the problem type
4. For minimization: check ``optimum="minimum"`` or negate the score


**How do I make results reproducible?**

Set the random seed:

.. code-block:: python

    opt = HillClimbingOptimizer(search_space, random_state=42)
