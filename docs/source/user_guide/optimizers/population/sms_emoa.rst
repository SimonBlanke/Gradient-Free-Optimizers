========
SMS-EMOA
========

SMS-EMOA (S-Metric Selection Evolutionary Multi-Objective Algorithm) is a
steady-state multi-objective optimizer that uses the hypervolume indicator
for environmental selection. Introduced by Beume, Naujoks and Emmerich
(2007), it selects against the individual whose removal causes the smallest
loss in dominated hypervolume.

The hypervolume indicator (also called S-metric) measures the volume of
objective space that is dominated by the current population and bounded by a
reference point. Among all unary quality indicators for Pareto fronts, the
hypervolume is the only one that is strictly monotone with respect to Pareto
dominance. This theoretical property gives SMS-EMOA the strongest convergence
guarantee of the three multi-objective algorithms in this library.

The steady-state design processes one offspring per iteration. After
evaluating the offspring, the combined population (size N+1) is ranked by
non-dominated sorting, and the individual with the smallest hypervolume
contribution in the worst front is removed.


Algorithm
---------

Each iteration follows these steps:

1. **Tournament selection**: Pick two parents from the population based on
   scalar fitness.
2. **Crossover + Mutation**: Produce one offspring via uniform crossover and
   self-adaptive hill-climbing mutation.
3. **Evaluation**: Compute the offspring's objective vector.
4. **Combine**: Add the offspring to the population (now size N+1).
5. **Non-dominated sort**: Rank the combined set into fronts.
6. **Remove worst**: In the last (worst) front, compute each member's
   exclusive hypervolume contribution. Remove the member with the smallest
   contribution. Population returns to size N.

.. code-block:: text

    Initialize population P of size N
    Evaluate objectives for each individual

    For each iteration:
        parent_a, parent_b = tournament_select(P)
        child = crossover(parent_a, parent_b) + mutation
        F(child) = evaluate(child)

        R = P + {child}                      (size N+1)
        Fronts = non_dominated_sort(R)
        F_worst = Fronts[-1]

        if |F_worst| == 1:
            remove that individual
        else:
            ref = compute_reference_point(F_worst)
            contributions = hypervolume_contributions(F_worst, ref)
            remove individual with min(contributions)

        P = R \ {removed}                    (back to size N)

.. note::

    The hypervolume contribution of a point is the volume of objective
    space that only that point dominates. Removing a point with large
    contribution would significantly reduce the overall dominated
    hypervolume, so SMS-EMOA protects it. Points in crowded regions
    tend to have small contributions, so they are removed preferentially.
    This produces a similar diversity effect to NSGA-II's crowding
    distance, but with a mathematically stronger foundation.


Hypervolume Computation
^^^^^^^^^^^^^^^^^^^^^^^

The hypervolume is computed relative to a reference point derived from the
worst front. For 2 objectives, the computation runs in O(n log n) time
using a sweep-line algorithm. For 3 or more objectives, the library uses
a recursive slicing approach (HSO algorithm) that is exact but becomes
expensive for large fronts with many objectives.

For problems with 4+ objectives and large populations, the hypervolume
computation may become the bottleneck. In such cases, consider using
MOEA/D instead, which has O(1) per-subproblem overhead regardless of
the number of objectives.


Parameters
----------

.. list-table::
    :header-rows: 1
    :widths: 20 15 15 50

    * - Parameter
      - Type
      - Default
      - Description
    * - ``population``
      - int
      - 20
      - Population size. Kept constant through steady-state replacement.
    * - ``crossover_rate``
      - float
      - 0.9
      - Probability of creating offspring via crossover.


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import SMSEMOAOptimizer

    def bi_objective(params):
        x = params["x"]
        y = params["y"]
        f1 = -(x**2 + y**2)
        f2 = -((x - 3)**2 + (y - 3)**2)
        return [f1, f2]

    search_space = {
        "x": np.linspace(-5, 5, 50),
        "y": np.linspace(-5, 5, 50),
    }

    opt = SMSEMOAOptimizer(search_space, population=30)
    opt.search(bi_objective, n_iter=300, n_objectives=2, verbosity=False)

    pareto = opt.pareto_front
    print(pareto[["x", "y", "objective_0", "objective_1"]])


When to Use
-----------

**Good for:**

- 2-3 objective problems where you want the best possible hypervolume
- Situations where theoretical convergence guarantees matter
- Online optimization where you want the population to improve after
  every single evaluation (steady-state)

**Not ideal for:**

- Many-objective problems (4+ objectives) due to hypervolume computation cost
- Very large populations combined with many objectives
- Problems where uniform Pareto front coverage is more important than
  hypervolume (use MOEA/D instead)

**Compared to NSGA-II**, SMS-EMOA replaces crowding distance with
hypervolume contribution, which provides a mathematically stronger
selection signal. In practice, both produce similar results for
well-behaved 2D Pareto fronts, but SMS-EMOA handles irregular front
shapes more robustly.

**Compared to MOEA/D**, SMS-EMOA does not require weight vector
generation or neighborhood structures. It adapts purely based on
the hypervolume of the current population. However, MOEA/D's
decomposition approach scales better to many objectives.


Trade-offs
----------

- Steady-state replacement (one per iteration) means the population is
  always up-to-date, but each evaluation triggers a full non-dominated
  sort and hypervolume computation.
- For 2 objectives, the overhead is negligible (O(n log n) per iteration).
  For 3 objectives, it grows to O(n^2 log n). Beyond that, it becomes the
  dominant cost.
- The reference point for hypervolume is computed from the worst front,
  not the entire population. This makes the selection robust to outliers
  but means the reference point can shift between iterations.


Related Algorithms
------------------

- :doc:`nsga2` uses crowding distance instead of hypervolume for
  diversity preservation. Faster per iteration, weaker theoretical
  guarantees.
- :doc:`moead` decomposes the problem into scalar subproblems.
  Scales better to many objectives and produces uniformly distributed
  fronts, but requires weight vector configuration.
- :doc:`genetic_algorithm` shares the evolutionary operators but
  works on a single scalar objective.
