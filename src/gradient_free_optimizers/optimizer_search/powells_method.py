# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Powell's method using sequential line searches along conjugate directions."""

from typing import Literal

from ..optimizers import PowellsMethod as _PowellsMethod
from ..search import Search


class PowellsMethod(_PowellsMethod, Search):
    """
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
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the search process.
    epsilon : float
        The step-size for hill climbing line search. Only used when
        line_search="hill_climb".
    distribution : str
        The type of distribution to sample from for hill climbing line search.
        Options are "normal", "laplace", "gumbel", or "logistic".
    n_neighbours : int
        The number of neighbours to sample and evaluate during hill climbing
        line search.
    iters_p_dim : int
        Number of evaluations per direction during line search. Higher values
        provide more accurate line searches but increase function evaluations.
        Default is 10.
    line_search : str
        Line search method to use. Options are "grid" (systematic evaluation),
        "golden" (golden-section search), or "hill_climb" (local search).
        Default is "grid".
    convergence_threshold : float
        Minimum total improvement per cycle required to continue. If the
        sum of improvements across all directions falls below this threshold,
        the optimizer considers itself converged and switches to random
        exploration. Default is 1e-8.

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
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        epsilon: float = 0.03,
        distribution: str = "normal",
        n_neighbours: int = 3,
        iters_p_dim: int = 10,
        line_search: Literal["grid", "golden", "hill_climb"] = "grid",
        convergence_threshold: float = 1e-8,
    ):
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
