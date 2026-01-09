# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Parallel tempering using multiple annealers with temperature swapping."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
)
from ..search import Search


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, Search):
    """
    Ensemble of simulated annealers at different temperatures with periodic swapping.

    Parallel Tempering (also known as Replica Exchange Monte Carlo) runs multiple
    simulated annealing processes simultaneously at different temperatures. The
    key innovation is periodic swapping of temperatures between the parallel
    searches based on the Metropolis criterion. This allows solutions found at
    high temperatures (broad exploration) to be refined at low temperatures
    (focused exploitation), and vice versa.

    The algorithm maintains a population of searchers, each operating at a
    different temperature. Periodically, adjacent temperature levels attempt to
    swap their current positions. This mechanism helps overcome energy barriers
    that would trap a single simulated annealing run, making the algorithm
    particularly effective for rugged optimization landscapes.

    The algorithm is well-suited for:

    - Highly multimodal optimization problems
    - Problems with deep local optima that trap single-searcher methods
    - Scenarios where computational resources allow parallel evaluations
    - Sampling from complex probability distributions

    The `population` parameter controls the number of parallel searchers, while
    `n_iter_swap` determines how frequently temperature swaps are attempted.

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
    population : int
        The number of parallel simulated annealers running at different
        temperatures. More searchers provide better coverage but require
        more function evaluations. Default is 5.
    n_iter_swap : int
        Number of iterations between temperature swap attempts. Lower values
        increase communication between temperature levels. Default is 5.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import ParallelTemperingOptimizer

    >>> def griewank(para):
    ...     x, y = para["x"], para["y"]
    ...     sum_sq = (x**2 + y**2) / 4000
    ...     prod_cos = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    ...     return -(sum_sq - prod_cos + 1)

    >>> search_space = {
    ...     "x": np.linspace(-600, 600, 1000),
    ...     "y": np.linspace(-600, 600, 1000),
    ... }

    >>> opt = ParallelTemperingOptimizer(search_space, population=10, n_iter_swap=10)
    >>> opt.search(griewank, n_iter=2000)
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
        population: int = 5,
        n_iter_swap: int = 5,
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
            population=population,
            n_iter_swap=n_iter_swap,
        )
