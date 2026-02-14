# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Downhill simplex (Nelder-Mead) using geometric simplex transformations."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import DownhillSimplexOptimizer as _DownhillSimplexOptimizer
from ..search import Search


class DownhillSimplexOptimizer(_DownhillSimplexOptimizer, Search):
    r"""
    Derivative-free optimizer using geometric simplex transformations.

    The Downhill Simplex method (also known as Nelder-Mead) is a classic
    derivative-free optimization algorithm that maintains a simplex of n+1 points
    in n-dimensional space. The algorithm iteratively transforms this simplex
    through reflection, expansion, contraction, and shrinking operations to
    move towards better regions of the search space.

    At each iteration, the worst point of the simplex is identified and replaced
    through one of four operations: reflection (moving away from the worst point),
    expansion (extending further in a promising direction), contraction (pulling
    back toward the centroid), or shrinking (contracting the entire simplex toward
    the best point). This adaptive behavior allows efficient navigation without
    gradient information.

    The algorithm is well-suited for:

    - Low to moderate dimensional problems (typically < 20 dimensions)
    - Smooth objective functions without too many local optima
    - Problems where derivatives are unavailable or expensive to compute
    - Initial exploration before applying gradient-based methods

    The simplex parameters (alpha, gamma, beta, sigma) control the aggressiveness
    of each transformation. The default values are well-established and work
    well for most problems.

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
    alpha : float, default=1
        Reflection coefficient. Controls how far the reflected point is
        placed from the centroid, opposite to the worst vertex. Standard
        value is 1.0.

        - ``0.5-0.8``: Conservative reflection, smaller steps.
        - ``1.0``: Standard reflection distance (default).
        - ``1.5-2.0``: Aggressive reflection, larger exploratory steps.

    gamma : float, default=2
        Expansion coefficient. When reflection yields a new best point,
        the algorithm extends further in that direction by this factor.

        - ``1.5``: Mild expansion.
        - ``2.0``: Standard expansion (default).
        - ``3.0+``: Aggressive expansion, risky but fast for smooth
          functions.

        Expansion only occurs when the reflected point is better than all
        current simplex vertices.

    beta : float, default=0.5
        Contraction coefficient. When reflection does not improve the
        simplex, the worst point is pulled toward the centroid by this
        factor. Values between 0 and 1.

        - ``0.25``: Strong contraction, aggressive shrinking.
        - ``0.5``: Standard contraction (default).
        - ``0.75``: Gentle contraction, slower convergence.

    sigma : float, default=0.5
        Shrink coefficient. When contraction also fails, the entire simplex
        shrinks toward the best vertex by this factor.

        - ``0.25``: Aggressive shrinking, fast convergence but may lose
          coverage.
        - ``0.5``: Standard shrinking (default).
        - ``0.75``: Gentle shrinking, maintains more coverage.

        Shrinking is a last resort operation that reduces the simplex
        volume, typically indicating the optimizer is converging.

    Notes
    -----
    The Nelder-Mead algorithm maintains a simplex of n+1 vertices in
    n-dimensional space. At each iteration, it applies one of four
    operations to the worst vertex :math:`x_w`:

    1. **Reflection**: :math:`x_r = (1 + \\alpha) \\bar{x} - \\alpha x_w`
    2. **Expansion**: :math:`x_e = (1 + \\gamma) \\bar{x} - \\gamma x_w`
       (if reflection was the new best)
    3. **Contraction**: :math:`x_c = (1 - \\beta) \\bar{x} + \\beta x_w`
       (if reflection was worst)
    4. **Shrink**: All vertices move toward the best vertex by factor
       ``sigma``

    where :math:`\\bar{x}` is the centroid of all vertices except the worst.

    The algorithm requires n+1 initial points to form the simplex. It does
    not use gradient information and typically converges in O(n^2)
    iterations for smooth functions.

    For visual explanations and tuning guides, see
    the :ref:`Downhill Simplex user guide <downhill_simplex>`.

    See Also
    --------
    PowellsMethod : Another derivative-free method using sequential line searches.
    PatternSearch : Direct search using geometric patterns instead of a simplex.
    HillClimbingOptimizer : Simpler local search without simplex geometry.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import DownhillSimplexOptimizer

    >>> def rosenbrock(para):
    ...     x, y = para["x"], para["y"]
    ...     return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-2, 2, 100),
    ...     "y": np.linspace(-1, 3, 100),
    ... }

    >>> opt = DownhillSimplexOptimizer(search_space)
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
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        alpha: float = 1,
        gamma: float = 2,
        beta: float = 0.5,
        sigma: float = 0.5,
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
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
        )
