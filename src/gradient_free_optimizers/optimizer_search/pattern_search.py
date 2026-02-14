# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Pattern search using geometric patterns around the current best point."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import PatternSearch as _PatternSearch
from ..search import Search


class PatternSearch(_PatternSearch, Search):
    r"""
    Direct search optimizer using geometric patterns around the current best point.

    Pattern Search (also known as Coordinate Search or Compass Search) is a
    derivative-free optimization method that explores the search space using
    a fixed geometric pattern of points around the current best solution.
    At each iteration, the algorithm evaluates all points in the pattern and
    moves to the best one found. If no improvement is found, the pattern size
    is reduced, allowing for finer local search.

    The pattern typically consists of points arranged symmetrically around
    the current position, such as along coordinate axes or in a more complex
    geometric arrangement. This systematic exploration ensures the algorithm
    does not miss improvements in any direction.

    The algorithm is well-suited for:

    - Problems with discontinuous or noisy objective functions
    - Situations requiring guaranteed descent (never accepts worse solutions)
    - Low-dimensional optimization problems
    - Problems where simplicity and robustness are valued over speed

    The `pattern_size` parameter controls the initial search radius as a fraction
    of the search space. The `reduction` factor determines how quickly the
    pattern shrinks when stuck, trading off between exploration and convergence.

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

    n_positions : int, default=4
        Number of positions in the search pattern around the current best
        point. More positions provide better directional coverage but
        require more function evaluations per iteration.

        - ``2*d``: Common choice (one positive and negative step per
          dimension), where d is the number of dimensions.
        - ``4``: Good default for low-dimensional problems.
        - ``8-16``: Better coverage for higher-dimensional problems.

    pattern_size : float, default=0.25
        Initial pattern size as a fraction of each dimension's range.
        Determines the initial search radius around the current best point.

        - ``0.05-0.1``: Small initial pattern, fine local search from the
          start.
        - ``0.2-0.3``: Moderate initial range (default region).
        - ``0.5-1.0``: Large initial pattern, broad exploration.

        Example: For a dimension ``np.linspace(0, 100, 1000)``,
        ``pattern_size=0.25`` means the initial pattern spans ~25 units.

    reduction : float, default=0.9
        Factor by which the pattern size is reduced when no improvement is
        found. Applied multiplicatively: ``new_size = pattern_size * reduction``.

        - ``0.5``: Aggressive reduction, fast convergence but may skip
          regions.
        - ``0.9``: Gentle reduction, thorough coverage (default).
        - ``0.95-0.99``: Very slow reduction, extensive exploration at
          each scale.

    Notes
    -----
    Pattern Search evaluates a fixed geometric pattern of points around
    the current best position:

    1. Generate ``n_positions`` points arranged symmetrically around the
       current best.
    2. Evaluate all pattern points and move to the best if it improves.
    3. If no improvement is found, reduce pattern size by ``reduction``.

    The pattern size after k consecutive failures is:

    .. math::

        s_k = s_0 \\cdot r^k

    where :math:`s_0` = ``pattern_size`` and :math:`r` = ``reduction``.

    Pattern search provides guaranteed descent (never accepts worse
    solutions) and converges to a stationary point under mild conditions.

    For visual explanations and tuning guides, see
    the :ref:`Pattern Search user guide <pattern_search>`.

    See Also
    --------
    PowellsMethod : Line-search-based optimization along conjugate directions.
    DownhillSimplexOptimizer : Adaptive simplex-based search without fixed patterns.
    DirectAlgorithm : Adaptive subdivision that automatically refines promising regions.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import PatternSearch

    >>> def step_function(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(np.floor(x) + np.floor(y))

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = PatternSearch(search_space, n_positions=8, pattern_size=0.3)
    >>> opt.search(step_function, n_iter=200)
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
        n_positions=4,
        pattern_size=0.25,
        reduction=0.9,
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
            n_positions=n_positions,
            pattern_size=pattern_size,
            reduction=reduction,
        )
