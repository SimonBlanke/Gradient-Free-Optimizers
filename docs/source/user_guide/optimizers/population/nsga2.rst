======
NSGA-II
======

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective
optimizer that evolves a population toward the Pareto front of a problem
with two or more conflicting objectives. It was introduced by Deb et al.
(2002) and remains one of the most widely used multi-objective evolutionary
algorithms.

The algorithm combines non-dominated sorting with a crowding distance metric.
Non-dominated sorting assigns each solution a rank based on how many other
solutions dominate it. Within the same rank, crowding distance measures how
isolated a solution is from its neighbors in objective space. Selection
favors lower ranks and, within a rank, solutions in less crowded regions.
This dual pressure drives the population toward a well-distributed
Pareto front.


Algorithm
---------

Each generation follows these steps:

1. **Tournament selection**: Pick two random individuals. Prefer the one
   with lower non-dominated rank. If ranks tie, prefer the one with
   higher crowding distance.
2. **Crossover**: Uniform crossover between two selected parents, producing
   one offspring.
3. **Mutation**: Self-adaptive hill-climbing perturbation on the offspring.
4. **Evaluation**: Evaluate the offspring's objective vector.
5. **Environmental selection** (after ``population`` evaluations): Combine
   the parent population and all offspring. Non-dominated sort the combined
   set, then fill the new population front-by-front. The last front that
   does not fully fit is trimmed using crowding distance.

.. code-block:: text

    Initialize population P of size N
    Evaluate objectives for each individual

    For each generation:
        Q = empty offspring set
        For i = 1 to N:
            parent_a = binary_tournament(P)
            parent_b = binary_tournament(P)
            child = crossover(parent_a, parent_b) + mutation
            Evaluate child
            Q = Q + {child}

        R = P + Q                         (size 2N)
        Fronts = non_dominated_sort(R)
        P_new = empty

        For each front F in Fronts:
            if |P_new| + |F| <= N:
                P_new = P_new + F
            else:
                Compute crowding_distance(F)
                Sort F by crowding distance (descending)
                P_new = P_new + F[1..(N - |P_new|)]
                break

        P = P_new

.. note::

    NSGA-II's selection pressure comes from two independent criteria:
    Pareto dominance determines rank, and crowding distance preserves
    spread. This means a dominated solution in a sparse region can
    survive over a non-dominated solution in a dense region when the
    final front is trimmed.


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
      - Number of individuals. Larger populations cover more of the
        Pareto front but require more evaluations per generation.
    * - ``crossover_rate``
      - float
      - 0.9
      - Probability of creating offspring via crossover rather than
        cloning a single parent.


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import NSGA2Optimizer

    def bi_objective(params):
        x = params["x"]
        y = params["y"]
        f1 = -(x**2 + y**2)             # minimize distance to origin
        f2 = -((x - 3)**2 + (y - 3)**2) # minimize distance to (3,3)
        return [f1, f2]

    search_space = {
        "x": np.linspace(-5, 5, 50),
        "y": np.linspace(-5, 5, 50),
    }

    opt = NSGA2Optimizer(search_space, population=30)
    opt.search(bi_objective, n_iter=300, n_objectives=2, verbosity=False)

    # The Pareto front contains all non-dominated solutions
    pareto = opt.pareto_front
    print(pareto[["x", "y", "objective_0", "objective_1"]])


When to Use
-----------

**Good for:**

- Problems with 2-3 conflicting objectives
- When you need a diverse set of trade-off solutions
- When the Pareto front shape is unknown ahead of time

**Not ideal for:**

- Single-objective problems (use a simpler algorithm)
- Many-objective problems (4+ objectives) where crowding distance
  degrades; consider NSGA-III or MOEA/D instead
- Very small evaluation budgets (population overhead)


Trade-offs
----------

- Larger ``population`` gives better Pareto front coverage but needs
  proportionally more iterations per generation.
- The crowding distance metric works well for 2-3 objectives but becomes
  less discriminative in higher dimensions.
- Generational replacement means the entire population is re-ranked after
  each generation, which adds overhead compared to steady-state approaches.


Related Algorithms
------------------

- :doc:`moead` uses decomposition into scalar subproblems instead of
  Pareto ranking. Often produces more uniformly distributed fronts.
- :doc:`genetic_algorithm` shares the crossover/mutation operators but
  operates on a single scalar objective.
- :doc:`differential_evolution` also evolves a population but uses
  difference vectors rather than crossover for reproduction.
