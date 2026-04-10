=======
MOEA/D
=======

MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) takes
a fundamentally different approach to multi-objective optimization than
Pareto-based algorithms like NSGA-II. Instead of ranking the entire population
by dominance, it decomposes the multi-objective problem into N scalar
subproblems, each defined by a weight vector. Every individual in the
population is responsible for one subproblem and only interacts with its
nearest neighbors in weight space.

Introduced by Zhang and Li (2007), MOEA/D is particularly effective at
producing uniformly distributed Pareto fronts because the weight vectors are
spread evenly across the objective space. The Tchebycheff scalarization used
by default can handle both convex and concave front shapes, unlike simpler
weighted-sum approaches.


Algorithm
---------

The setup phase creates the decomposition structure:

1. Generate N uniformly distributed weight vectors on the unit simplex
   (Das-Dennis lattice design).
2. For each weight vector, find its T nearest neighbors by Euclidean
   distance. These neighborhoods define who can share solutions.
3. Initialize a reference point z* with the best observed value for each
   objective.

Each iteration then works on one subproblem:

1. **Select subproblem**: Cycle through subproblems round-robin.
2. **Select parents**: Pick two parents randomly from the current
   subproblem's neighborhood.
3. **Crossover + Mutation**: Uniform crossover followed by self-adaptive
   hill-climbing mutation.
4. **Evaluate**: Compute the offspring's objective vector.
5. **Update reference point**: If the offspring improves any component
   of z*, update it.
6. **Replace neighbors**: For each neighbor j, if the offspring has
   better Tchebycheff fitness for subproblem j, replace j's solution
   with the offspring.

.. code-block:: text

    Generate weight vectors lambda_1 ... lambda_N
    Compute neighborhoods B(i) for each i
    Initialize reference point z*

    For each iteration:
        i = current subproblem (round-robin)
        Select k, l randomly from B(i)
        child = crossover(x_k, x_l) + mutation
        F(child) = evaluate(child)

        Update z*: z*_j = max(z*_j, F_j(child)) for each j

        For each j in B(i):
            g_new = tchebycheff(F(child), lambda_j, z*)
            g_old = tchebycheff(F(x_j), lambda_j, z*)
            if g_new is better:
                x_j = child

.. note::

    The immediate replacement is a defining feature of MOEA/D. When an
    offspring replaces a neighbor, subsequent subproblems in the same
    generation already see the updated solution. This propagation effect
    accelerates convergence compared to generational algorithms where
    replacements only take effect in the next generation.


Tchebycheff Scalarization
^^^^^^^^^^^^^^^^^^^^^^^^^

Each subproblem converts the multi-objective problem to a scalar using
Tchebycheff decomposition. For a weight vector lambda and reference
point z*, the scalarized fitness is:

.. code-block:: text

    fitness(x) = -max_j { lambda_j * (z*_j - f_j(x)) }

The solution that minimizes the worst weighted gap to the ideal point
gets the highest fitness. Different weight vectors focus on different
regions of the Pareto front, ensuring coverage.


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
      - Number of subproblems (and weight vectors). For 2 objectives,
        this directly determines front resolution. For 3+ objectives, the
        actual count may be adjusted slightly to fit the simplex lattice.
    * - ``n_neighbors``
      - int or None
      - None
      - Neighborhood size T for each weight vector. Defaults to
        ``max(3, population // 5)``. Smaller values focus updates
        locally (faster convergence, risk of premature convergence).
        Larger values spread updates more broadly (better diversity,
        slower convergence).
    * - ``crossover_rate``
      - float
      - 0.9
      - Probability of crossover vs. cloning a parent.


Neighborhood Size
^^^^^^^^^^^^^^^^^

The neighborhood parameter controls the trade-off between local exploitation
and global exploration. With ``n_neighbors=3``, each subproblem only shares
solutions with its two closest neighbors, leading to tight clusters along
the Pareto front. With ``n_neighbors=population``, every subproblem
interacts with everyone, behaving more like a global evolutionary algorithm.

.. code-block:: python

    # Tight neighborhoods - fast convergence, may miss parts of the front
    opt = MOEADOptimizer(search_space, population=30, n_neighbors=3)

    # Broader neighborhoods - slower but more diverse
    opt = MOEADOptimizer(search_space, population=30, n_neighbors=10)


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import MOEADOptimizer

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

    opt = MOEADOptimizer(search_space, population=30)
    opt.search(bi_objective, n_iter=300, n_objectives=2, verbosity=False)

    pareto = opt.pareto_front
    print(pareto[["x", "y", "objective_0", "objective_1"]])


When to Use
-----------

**Good for:**

- Problems with 2-3 objectives where you need a well-distributed front
- Convex or concave Pareto front shapes (Tchebycheff handles both)
- Situations where uniform coverage of the trade-off surface matters
  more than finding the extreme points

**Not ideal for:**

- Single-objective problems
- Problems where the number of objectives changes during optimization
- Very small populations (the weight vector decomposition needs
  enough subproblems to cover the front)

**Compared to NSGA-II**, MOEA/D tends to produce more evenly spaced
Pareto fronts because the weight vectors are pre-distributed across the
objective space. NSGA-II's crowding distance can leave gaps in the front
that it has no mechanism to fill proactively.


Three-Objective Example
-----------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import MOEADOptimizer

    def tri_objective(params):
        x = params["x"]
        y = params["y"]
        return [
            -(x**2 + y**2),
            -((x - 3)**2 + y**2),
            -(x**2 + (y - 3)**2),
        ]

    search_space = {
        "x": np.linspace(-5, 5, 50),
        "y": np.linspace(-5, 5, 50),
    }

    opt = MOEADOptimizer(search_space, population=50)
    opt.search(tri_objective, n_iter=500, n_objectives=3, verbosity=False)

    pf = opt.pareto_front
    print(f"Pareto front: {len(pf)} solutions")
    print(pf[["x", "y", "objective_0", "objective_1", "objective_2"]].head())


Trade-offs
----------

- ``population`` directly controls Pareto front resolution. More
  weight vectors means finer granularity but more evaluations per cycle.
- ``n_neighbors`` balances convergence speed against diversity. The
  default (20% of population) works well in most cases.
- Tchebycheff scalarization needs a reference point that improves during
  the search. Early iterations may have unstable rankings as z* shifts.


Related Algorithms
------------------

- :doc:`nsga2` uses Pareto ranking and crowding distance instead of
  decomposition. Better when you do not need uniform front coverage.
- :doc:`genetic_algorithm` shares the crossover/mutation operators but
  works on a single scalar objective.
- :doc:`evolution_strategy` also uses self-adaptive mutation but without
  the decomposition structure.
