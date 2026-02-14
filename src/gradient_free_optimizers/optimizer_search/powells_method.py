# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Powell's method using sequential line searches along conjugate directions."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import PowellsMethod as _PowellsMethod
from ..search import Search


class PowellsMethod(_PowellsMethod, Search):
    r"""
    Derivative-free optimizer using sequential line searches along conjugate directions.

    Powell's method is a powerful derivative-free optimization algorithm that
    performs sequential one-dimensional line searches along a set of directions.
    Starting with the coordinate axes as initial search directions, the algorithm
    updates these directions after each complete cycle to form conjugate directions.
    This strategy leads to faster convergence than simple coordinate descent,
    particularly for functions with elliptical contours.

    The algorithm proceeds in cycles: during each cycle, it performs a line search
    along each direction, finding the optimal step in that direction. After
    completing all directions, it updates the direction set by replacing one
    direction with the overall displacement vector. This builds up information
    about the function's curvature without requiring explicit gradient or Hessian
    computation.

    The algorithm is well-suited for:

    - Smooth, unimodal objective functions
    - Problems where function evaluations are expensive (efficient per iteration)
    - Moderate dimensional problems (scales reasonably up to ~50 dimensions)
    - Situations where gradient-based methods cannot be applied

    Multiple line search strategies are available: "grid" for systematic evaluation,
    "golden" for golden-section search, and "hill_climb" for local optimization.

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
        one becomes the starting point for the optimizer.

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

        This is especially useful for local search optimizers (Hill Climbing,
        Simulated Annealing) that can get trapped. For population-based
        optimizers, the effect is less pronounced since they already maintain
        diversity through multiple agents.

    epsilon : float, default=0.03
        Step size for hill climbing line search. Only used when
        ``line_search="hill_climb"``. Ignored for "grid" and "golden"
        line search methods.

    distribution : {"normal", "laplace", "gumbel", "logistic"}, default="normal"
        Distribution for sampling during hill climbing line search. Only
        used when ``line_search="hill_climb"``. Ignored for other line
        search methods.

    iters_p_dim : int, default=10
        Number of function evaluations per direction during each line
        search. Higher values provide more accurate line searches but
        increase the total cost per cycle.

        - ``5``: Fast but coarse line search.
        - ``10``: Good balance for most problems (default).
        - ``20-50``: Thorough line search, recommended for functions with
          narrow valleys.

        Total evaluations per cycle is approximately
        ``iters_p_dim * d`` where d is the number of dimensions.

    line_search : {"grid", "golden", "hill_climb"}, default="grid"
        Method used for one-dimensional optimization along each direction.

        - ``"grid"``: Evaluates evenly spaced points along the direction.
          Simple and robust, no assumptions about function shape.
        - ``"golden"``: Golden-section search, efficient for unimodal
          functions along each direction.
        - ``"hill_climb"``: Uses hill climbing with ``epsilon`` and
          ``distribution`` for the line search. More flexible but
          stochastic.

    convergence_threshold : float, default=1e-8
        Minimum total improvement per cycle required to continue
        optimizing. If the sum of improvements across all direction
        searches falls below this threshold, the optimizer considers
        itself converged and switches to random exploration.

        - ``1e-6``: Relatively loose convergence criterion.
        - ``1e-8``: Standard precision (default).
        - ``1e-12``: Very tight convergence, for high-precision needs.

    Notes
    -----
    Powell's method performs sequential line searches along a set of
    d conjugate directions:

    1. Start with coordinate axes as initial directions.
    2. For each direction :math:`\\mathbf{d}_i`, find the optimal step
       :math:`\\alpha_i` via line search.
    3. After completing all directions, update the direction set by
       replacing one direction with the overall displacement vector.
    4. If total improvement falls below ``convergence_threshold``, switch
       to random exploration.

    The direction update builds up curvature information, creating
    conjugate directions that accelerate convergence for quadratic-like
    functions. For a perfectly quadratic function in d dimensions, the
    algorithm converges in d cycles.

    For visual explanations and tuning guides, see
    the :ref:`Powell's Method user guide <powells_method>`.

    See Also
    --------
    DownhillSimplexOptimizer : Derivative-free optimization using simplex geometry.
    PatternSearch : Simpler pattern-based direct search method.
    HillClimbingOptimizer : Stochastic local search without line search structure.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import PowellsMethod

    >>> def quadratic(para):
    ...     x, y, z = para["x"], para["y"], para["z"]
    ...     return -(x**2 + 2*y**2 + 3*z**2 + x*y + y*z)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ...     "z": np.linspace(-10, 10, 100),
    ... }

    >>> opt = PowellsMethod(search_space, iters_p_dim=15, line_search="golden")
    >>> opt.search(quadratic, n_iter=300)
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        epsilon: float = 0.03,
        distribution: str = "normal",
        n_neighbours: int = 3,  # no-op, kept for backwards compatibility
        iters_p_dim: int = 10,
        line_search: Literal["grid", "golden", "hill_climb"] = "grid",
        convergence_threshold: float = 1e-8,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
            iters_p_dim=iters_p_dim,
            line_search=line_search,
            convergence_threshold=convergence_threshold,
        )
