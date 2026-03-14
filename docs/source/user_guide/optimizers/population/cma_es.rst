======
CMA-ES
======

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a state-of-the-art
evolutionary algorithm for continuous optimization. It maintains a multivariate
normal distribution over the search space and adapts a full covariance matrix to
learn the correlation structure of the fitness landscape. Each generation, the
algorithm samples candidate solutions, ranks them by fitness, shifts the
distribution mean toward the best solutions, and updates the covariance matrix
using evolution paths. A cumulative step-size adaptation mechanism controls the
global step size.

CMA-ES is widely regarded as the default algorithm for continuous black-box
optimization in moderate dimensions (up to ~100). Unlike simpler evolution
strategies that use only a scalar or diagonal step-size, CMA-ES learns
arbitrary axis-aligned and rotated ellipsoidal distributions. This makes it
particularly effective when parameters are correlated or have very different
sensitivities. For example, if increasing ``x`` should be accompanied by
decreasing ``y`` to improve the objective, CMA-ES will learn this relationship
and sample accordingly.

For mixed search spaces with discrete or categorical dimensions, the
implementation samples in a normalized continuous space and maps back to valid
grid values via rounding. This is the standard MI-CMA-ES approach.


Algorithm
---------

Each generation:

1. **Sample**: Draw ``population`` candidates from :math:`\mathcal{N}(m, \sigma^2 C)`
2. **Evaluate**: Score all candidates
3. **Rank**: Sort by fitness, select the best ``mu``
4. **Update mean**: Shift ``m`` toward the weighted mean of the best ``mu``
5. **Update evolution paths**: Accumulate step information (p_sigma, p_c)
6. **Update covariance**: Rank-one update (from p_c) + rank-mu update (from selected solutions)
7. **Adapt step size**: Increase sigma if steps are correlated, decrease if oscillating

.. code-block:: text

    x_k = mean + sigma * B @ D @ z_k    # sample (z_k ~ N(0, I))
    mean_new = sum(w_i * x_i:mu)         # weighted recombination
    p_sigma = (1-c_s) * p_sigma + ...    # evolution path for step size
    p_c = (1-c_c) * p_c + ...            # evolution path for covariance
    C = (1-c_1-c_mu) * C + c_1 * p_c @ p_c.T + c_mu * rank_mu_update
    sigma = sigma * exp(c_s/d_s * (||p_sigma||/E||N(0,I)|| - 1))

The covariance matrix ``C`` is decomposed as :math:`C = B D^2 B^T` where ``B``
holds the eigenvectors (rotation) and ``D`` the eigenvalues (axis lengths).

.. note::

    CMA-ES automatically sets most internal parameters (learning rates,
    weights, damping) from the dimensionality and population size. You
    typically only need to set ``population``, ``sigma``, and optionally
    ``ipop_restart``.


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
      - int | None
      - None
      - Candidates per generation (lambda). ``None`` uses ``4 + floor(3 * ln(n))``.
    * - ``mu``
      - int | None
      - None
      - Number of parents selected. ``None`` uses ``population // 2``.
    * - ``sigma``
      - float
      - 0.3
      - Initial step size as fraction of normalized space.
    * - ``ipop_restart``
      - bool
      - False
      - Enable IPOP restart on stagnation (doubles population each restart).


Step Size (sigma)
^^^^^^^^^^^^^^^^^

The initial sigma controls the initial spread of samples. CMA-ES adapts it
automatically, so the starting value is not critical.

.. code-block:: python

    # Conservative start (fine-tuning around a known good region)
    opt = CMAESOptimizer(search_space, sigma=0.1)

    # Broad initial exploration
    opt = CMAESOptimizer(search_space, sigma=0.5)


IPOP Restart
^^^^^^^^^^^^

When stagnation is detected, IPOP restarts with a doubled population and a new
random starting point. This is effective for multi-modal landscapes where
a single run may converge to a suboptimal local optimum.

.. code-block:: python

    opt = CMAESOptimizer(
        search_space,
        ipop_restart=True,
    )


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import CMAESOptimizer

    def rosenbrock(para):
        x, y = para["x"], para["y"]
        return -(100 * (y - x**2)**2 + (1 - x)**2)

    search_space = {
        "x": np.linspace(-5, 5, 1000),
        "y": np.linspace(-5, 5, 1000),
    }

    opt = CMAESOptimizer(
        search_space,
        population=20,
        sigma=0.3,
    )

    opt.search(rosenbrock, n_iter=500)
    print(f"Best: {opt.best_para}, Score: {opt.best_score}")


When to Use
-----------

**Good for:**

- Continuous optimization with correlated parameters
- Problems where parameter sensitivities differ strongly
- Moderate dimensionality (2-100 dimensions)
- Multi-modal landscapes (with ``ipop_restart=True``)

**Not ideal for:**

- Very high dimensions (>100), where the covariance matrix becomes expensive
- Purely discrete/combinatorial problems (GA or DE are better suited)
- Very tight iteration budgets (CMA-ES needs several generations to adapt)

**Compared to other population-based optimizers:**

- CMA-ES vs ES: CMA-ES adapts a full covariance matrix; ES uses scalar/diagonal step sizes
- CMA-ES vs PSO: CMA-ES models the landscape shape; PSO uses velocity/social dynamics
- CMA-ES vs DE: CMA-ES learns correlations explicitly; DE derives steps from population differences


High-Dimensional Example
-------------------------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import CMAESOptimizer

    def ellipsoid(para):
        total = 0
        for i, key in enumerate(sorted(para)):
            total += (10 ** (2 * i / 9)) * para[key] ** 2
        return -total

    search_space = {
        f"x{i}": np.linspace(-5, 5, 200)
        for i in range(10)
    }

    opt = CMAESOptimizer(
        search_space,
        population=30,
        sigma=0.3,
        ipop_restart=True,
    )

    opt.search(ellipsoid, n_iter=2000)
    print(f"Best score: {opt.best_score}")


Trade-offs
----------

- **Exploration vs. exploitation**: sigma controls initial spread; the algorithm
  self-adapts over time. IPOP restart adds macro-level exploration.
- **Computational overhead**: Per generation, CMA-ES performs an eigendecomposition
  of the covariance matrix (O(n^3)), making it expensive for very high dimensions.
- **Population size**: Larger populations improve robustness on multi-modal problems
  but require more evaluations per generation. The default heuristic is a good
  starting point.


Related Algorithms
------------------

- :doc:`evolution_strategy` - Simpler ES with mutation-based search
- :doc:`differential_evolution` - Self-adaptive step sizes from population differences
- :doc:`particle_swarm` - Swarm-based approach with velocity dynamics
