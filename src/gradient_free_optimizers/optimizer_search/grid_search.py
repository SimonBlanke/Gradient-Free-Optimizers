# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import GridSearchOptimizer as _GridSearchOptimizer


class GridSearchOptimizer(_GridSearchOptimizer, Search):
    """
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
    step_size : int
        The step size between grid points in index space. A step_size of 1
        evaluates every point, while larger values skip points for faster
        but coarser search. Default is 1.
    direction : str
        The traversal direction through the grid. "diagonal" moves across
        all dimensions simultaneously, while "orthogonal" proceeds along
        one axis at a time. Default is "diagonal".

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
        search_space: Dict[str, list],
        initialize: Dict[
            Literal["grid", "vertices", "random", "warm_start"],
            Union[int, list[dict]],
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        step_size: int = 1,
        direction: Literal["diagonal", "orthogonal"] = "diagonal",
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            step_size=step_size,
            direction=direction,
        )
