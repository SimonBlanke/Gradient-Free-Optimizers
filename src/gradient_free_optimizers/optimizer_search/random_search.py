# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Literal

from ..optimizers import RandomSearchOptimizer as _RandomSearchOptimizer
from ..search import Search


class RandomSearchOptimizer(_RandomSearchOptimizer, Search):
    """
    Simple optimizer that samples random positions from the search space.

    Random Search is the simplest optimization strategy that samples positions
    uniformly at random from the search space. Despite its simplicity, random
    search is surprisingly effective for many problems, especially in high
    dimensions where it can outperform grid search due to better coverage of
    important dimensions.

    The algorithm has no memory of previous evaluations and makes no assumptions
    about the objective function. Each iteration independently samples a random
    point, making it trivially parallelizable and resistant to getting stuck
    in local optima.

    The algorithm is well-suited for:

    - High-dimensional search spaces (avoids curse of dimensionality)
    - Baseline comparison for other optimization methods
    - Initial exploration before applying more sophisticated methods
    - Problems with many irrelevant dimensions
    - Embarrassingly parallel optimization scenarios

    Random search is often recommended as a first approach for hyperparameter
    optimization due to its simplicity and effectiveness. It provides uniform
    coverage of the search space without requiring any tuning parameters.

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

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RandomSearchOptimizer

    >>> def objective(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ... }

    >>> opt = RandomSearchOptimizer(search_space)
    >>> opt.search(objective, n_iter=1000)
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
        nth_process: int = None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            nth_process=nth_process,
        )
