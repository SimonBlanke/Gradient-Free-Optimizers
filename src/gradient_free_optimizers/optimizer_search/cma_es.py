# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""CMA-ES using covariance matrix adaptation for continuous domains."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    CMAESOptimizer as _CMAESOptimizer,
)
from ..search import Search


class CMAESOptimizer(_CMAESOptimizer, Search):
    """
    Evolutionary optimizer using covariance matrix adaptation.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
    state-of-the-art evolutionary algorithm for difficult continuous
    optimization problems. It adapts a full covariance matrix to learn
    the correlation structure of the fitness landscape, enabling
    efficient search even when parameters are strongly correlated or
    have different sensitivities.

    The algorithm maintains a multivariate normal distribution and
    iteratively:

    1. Samples ``population`` candidate solutions from the distribution
    2. Evaluates and ranks them by fitness
    3. Updates the distribution mean toward the best solutions
    4. Adapts the covariance matrix using evolution paths
    5. Controls the global step size via cumulative step-size adaptation

    CMA-ES is considered the gold standard for continuous black-box
    optimization. For mixed search spaces (discrete, categorical),
    this implementation samples in continuous space and rounds to the
    nearest valid value, which is a pragmatic compromise.

    The algorithm is well-suited for:

    - Continuous optimization with correlated parameters
    - Problems where parameter sensitivities differ strongly
    - Moderate dimensionality (up to ~100 dimensions)
    - Multi-modal landscapes (especially with IPOP restart)

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore, defined as a dictionary mapping parameter
        names to arrays of possible values.

        Each key is a parameter name (string), and each value is a numpy array
        or list of discrete values that the parameter can take. The optimizer
        will only evaluate positions that are on this discrete grid.

        Example: A 2D search space with 100 points per dimension::

            search_space = {
                "x": np.linspace(-10, 10, 100),
                "y": np.linspace(-10, 10, 100),
            }

        The resolution of each dimension (number of points in the array)
        directly affects optimization quality and speed. More points give
        finer resolution but increase the search space size exponentially.
    initialize : dict[str, int], default={"vertices": 4, "random": 2}
        Strategy for generating initial positions before the main optimization
        loop begins. Initialization samples are evaluated first, and the best
        one becomes the starting point (mean) for the CMA-ES distribution.

        Supported keys:

        - ``"grid"``: ``int`` -- Number of positions on a regular grid.
        - ``"vertices"``: ``int`` -- Number of corner/edge positions of the
          search space.
        - ``"random"``: ``int`` -- Number of uniformly random positions.
        - ``"warm_start"``: ``list[dict]`` -- Specific positions to evaluate,
          each as a dict mapping parameter names to values.

        Multiple strategies can be combined::

            initialize = {"vertices": 4, "random": 10}
            initialize = {"warm_start": [{"x": 0.5, "y": 1.0}], "random": 5}

        More initialization samples improve the starting point but consume
        iterations from ``n_iter``. For expensive objectives, a few targeted
        warm-start points are often more efficient than many random samples.
    constraints : list[callable], default=[]
        A list of constraint functions that restrict the search space. Each
        constraint is a callable that receives a parameter dictionary and
        returns ``True`` if the position is valid, ``False`` if it should
        be rejected.

        Rejected positions are discarded and regenerated: the optimizer
        resamples a new candidate position (up to 100 retries per step).
        During initialization, positions that violate constraints are
        filtered out entirely.

        Example: Constrain the search to a circular region::

            def circular_constraint(para):
                return para["x"]**2 + para["y"]**2 <= 25

            constraints = [circular_constraint]

        Multiple constraints are combined with AND logic (all must return
        ``True``).
    random_state : int or None, default=None
        Seed for the random number generator to ensure reproducible results.

        - ``None``: Use a new random state each run (non-deterministic).
        - ``int``: Seed the random number generator for reproducibility.

        Setting a fixed seed is recommended for debugging and benchmarking.
        Different seeds may lead to different optimization trajectories,
        especially for stochastic optimizers.
    rand_rest_p : float, default=0
        Probability of performing a random restart instead of the normal
        algorithm step. At each iteration, a uniform random number is drawn;
        if it falls below ``rand_rest_p``, the optimizer jumps to a random
        position instead of following its strategy.

        - ``0.0``: No random restarts (pure algorithm behavior).
        - ``0.01-0.05``: Light diversification, helps escape shallow local
          optima.
        - ``0.1-0.3``: Aggressive restarts, useful for highly multi-modal
          landscapes.
        - ``1.0``: Equivalent to random search.
    population : int or None, default=None
        Number of candidate solutions sampled per generation (lambda in
        CMA-ES notation). If ``None``, uses the standard heuristic:
        ``4 + floor(3 * ln(n_dimensions))``.

        - ``None``: Auto-compute based on dimensionality (recommended).
        - ``10-20``: Small populations for fast convergence on simple
          problems.
        - ``50-100``: Large populations for better exploration on
          multimodal or high-dimensional problems.

        Each generation requires ``population`` function evaluations,
        so total cost per generation scales linearly with this parameter.
    mu : int or None, default=None
        Number of best solutions selected as parents for the next
        generation. If ``None``, uses ``population // 2``.

        - ``None``: Auto-compute as half the population (recommended).
        - Smaller ``mu``: Stronger selection pressure, faster convergence
          but higher risk of premature convergence.
        - Larger ``mu``: Weaker selection pressure, better exploration.

        Must be less than or equal to ``population``.
    sigma : float, default=0.3
        Initial step size as a fraction of the normalized search space
        range. Controls the initial spread of sampled solutions around
        the mean.

        - ``0.1``: Conservative, tight initial sampling.
        - ``0.3``: Standard starting point (default).
        - ``0.5``: Broad initial exploration.

        CMA-ES adapts sigma automatically during optimization, so the
        initial value is not critical. Values between 0.1 and 0.5
        generally work well.
    ipop_restart : bool, default=False
        Enable IPOP (Increasing Population) restart strategy. When
        stagnation is detected (no improvement for many generations),
        the algorithm restarts with a doubled population size and a
        random starting point.

        - ``False``: No restarts, single run (default).
        - ``True``: Enable IPOP restarts for better global search on
          multimodal landscapes.

        IPOP-CMA-ES is particularly effective for problems with many
        local optima, as it combines the precision of CMA-ES with
        increasingly thorough global search.

    Notes
    -----
    CMA-ES adapts the search distribution using two evolution paths:

    - **Cumulation path for sigma** (p_sigma): Controls global step size
      via Cumulative Step-size Adaptation (CSA). If steps are correlated
      (consistent direction), sigma increases; if anti-correlated
      (oscillating), sigma decreases.
    - **Cumulation path for C** (p_c): Provides the rank-one update to
      the covariance matrix, capturing the dominant search direction.

    The covariance matrix is updated via:

    - **Rank-one update**: Uses p_c to learn the principal search
      direction.
    - **Rank-mu update**: Uses all mu selected solutions to learn the
      local landscape shape.

    For mixed search spaces (discrete/categorical dimensions), the
    algorithm operates in a normalized continuous space and maps back
    to valid values via rounding. This is a standard approach (MI-CMA-ES)
    that preserves the covariance adaptation while supporting non-continuous
    parameters.

    See Also
    --------
    EvolutionStrategyOptimizer : Simpler ES with self-adaptive sigma.
    DifferentialEvolutionOptimizer : DE using vector differences.
    ParticleSwarmOptimizer : Swarm intelligence approach.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import CMAESOptimizer

    >>> def rosenbrock(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(100 * (y - x**2)**2 + (1 - x)**2)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 1000),
    ...     "y": np.linspace(-5, 5, 1000),
    ... }

    >>> opt = CMAESOptimizer(search_space, population=20, sigma=0.3)
    >>> opt.search(rosenbrock, n_iter=500)
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        population: int = None,
        mu: int = None,
        sigma: float = 0.3,
        ipop_restart: bool = False,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []
        if conditions is None:
            conditions = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            conditions=conditions,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
            mu=mu,
            sigma=sigma,
            ipop_restart=ipop_restart,
        )
