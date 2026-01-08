# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Literal

from ..optimizers import LipschitzOptimizer as _LipschitzOptimizer
from ..search import Search


class LipschitzOptimizer(_LipschitzOptimizer, Search):
    """
    Global optimizer using Lipschitz continuity bounds for deterministic search.

    Lipschitz Optimization exploits the Lipschitz continuity property of objective
    functions to provide deterministic bounds on the optimum. If a function is
    Lipschitz continuous with constant L, the function value at any point cannot
    differ from nearby points by more than L times the distance. This property
    allows the algorithm to rule out regions that provably cannot contain the
    global optimum.

    The algorithm estimates the Lipschitz constant from observed data and uses
    it to compute upper bounds on the objective function across the search space.
    Points with the highest potential (according to these bounds) are selected
    for evaluation, gradually tightening the bounds until convergence.

    The algorithm is well-suited for:

    - Lipschitz continuous objective functions
    - Problems requiring global optimality guarantees
    - Low-dimensional optimization (bounds become loose in high dimensions)
    - Deterministic optimization without relying on random sampling

    This method provides theoretical guarantees for finding the global optimum
    but requires the objective function to satisfy Lipschitz continuity. The
    estimated Lipschitz constant adapts as more observations are collected.

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
        The probability of a random iteration during the the search process.
    warm_start_smbo : object, optional
        Previous SMBO state for warm starting optimization.
    max_sample_size : int
        Maximum number of candidate points for bound computation.
        Default is 10000000.
    sampling : dict
        Configuration for candidate sampling. Default is {"random": 1000000}.
    replacement : bool
        Whether to sample candidates with replacement. Default is True.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import LipschitzOptimizer

    >>> def smooth_function(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(x**2 + y**2)  # Lipschitz continuous

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = LipschitzOptimizer(search_space)
    >>> opt.search(smooth_function, n_iter=100)
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
        warm_start_smbo=None,
        max_sample_size: int = 10000000,
        sampling: dict[Literal["random"], int] = {"random": 1000000},
        replacement: bool = True,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )
