# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Grid search that systematically evaluates points across the search space."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import GridSearchOptimizer as _GridSearchOptimizer
from ..search import Search


class GridSearchOptimizer(_GridSearchOptimizer, Search):
    r"""
    Exhaustive search optimizer that systematically evaluates a grid of points.

    Grid Search is a brute-force optimization method that systematically
    evaluates the objective function at regularly spaced points across the
    search space. Starting from an initial position, the algorithm traverses
    the search space in a structured manner, evaluating every grid point within
    the specified step size. This guarantees finding the global optimum within
    the evaluated grid resolution.

    Unlike random or heuristic methods, grid search provides deterministic
    coverage of the search space. The direction parameter controls whether
    the search proceeds along coordinate axes (orthogonal) or across all
    dimensions simultaneously (diagonal).

    The algorithm is well-suited for:

    - Low-dimensional problems where exhaustive search is feasible
    - Situations requiring guaranteed coverage of the search space
    - Baseline comparisons for other optimization methods
    - Problems where the objective function is fast to evaluate

    The main limitation is the exponential growth of evaluations with
    dimensionality (curse of dimensionality). For high-dimensional problems,
    consider using random search or more sophisticated methods instead.

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

    step_size : int, default=1
        Step size between grid points in index space. A step_size of 1
        evaluates every point in the search space, while larger values
        skip points for faster but coarser coverage.

        - ``1``: Evaluate every grid point (exhaustive). Total evaluations
          = product of all dimension sizes.
        - ``2-5``: Skip points, reducing evaluations by ``step_size^d``
          where d is the number of dimensions.
        - ``10+``: Very coarse grid, fast but may miss optima in narrow
          regions.

        Example: For a 2D space with 100 points per dimension:

        - ``step_size=1`` evaluates 10,000 points
        - ``step_size=5`` evaluates ~400 points

    direction : {"diagonal", "orthogonal"}, default="diagonal"
        Traversal pattern through the grid.

        - ``"diagonal"``: Moves across all dimensions simultaneously,
          providing a more uniform coverage pattern. Generally recommended.
        - ``"orthogonal"``: Proceeds along one axis at a time (coordinate
          sweep). Can be more systematic for separable functions.

    resolution : int, default=100
        Number of grid points for continuous dimensions specified as
        tuples (e.g., ``(0.0, 10.0)``). These are automatically
        discretized into this many evenly-spaced points. Has no effect
        on dimensions already specified as arrays.

    Notes
    -----
    Grid search evaluates points on a regular grid defined by the search
    space arrays and ``step_size``. The total number of evaluations grows
    exponentially with dimensionality:

    .. math::

        N_{\\text{total}} = \\prod_{i=1}^{d} \\lceil |S_i| / \\text{step\\_size} \\rceil

    where :math:`|S_i|` is the number of values in dimension i. This
    "curse of dimensionality" makes grid search impractical for problems
    with more than a few dimensions.

    Grid search guarantees finding the global optimum within the grid
    resolution, but cannot find optima between grid points.

    For visual explanations and comparisons, see
    the :ref:`Grid Search user guide <grid_search>`.

    See Also
    --------
    RandomSearchOptimizer : Random sampling, often better in high dims.
    DirectAlgorithm : Adaptive grid subdivision that focuses on promising regions.
    PatternSearch : Systematic search using geometric patterns with adaptive resolution.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import GridSearchOptimizer

    >>> def objective(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 50),
    ...     "y": np.linspace(-5, 5, 50),
    ... }

    >>> # Evaluate every point in the grid
    >>> opt = GridSearchOptimizer(search_space, step_size=1)
    >>> opt.search(objective, n_iter=2500)
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
        step_size: int = 1,
        direction: Literal["diagonal", "orthogonal"] = "diagonal",
        resolution: int = 100,
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
            step_size=step_size,
            direction=direction,
            resolution=resolution,
        )
