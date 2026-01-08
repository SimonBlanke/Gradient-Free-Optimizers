# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random restart hill climbing with periodic restarts from random positions."""

from typing import Literal

from ..optimizers import (
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
)
from ..search import Search


class RandomRestartHillClimbingOptimizer(_RandomRestartHillClimbingOptimizer, Search):
    """
    Hill climbing variant that periodically restarts from random positions.

    Random Restart Hill Climbing addresses the local optima problem by periodically
    resetting the search to a new random position after a fixed number of iterations.
    This simple yet effective strategy allows the algorithm to explore multiple
    regions of the search space, increasing the probability of finding the global
    optimum. The best solution found across all restarts is retained.

    The algorithm is well-suited for:

    - Multimodal optimization problems with many local optima
    - Problems where the location of the global optimum is unknown
    - Scenarios where multiple independent searches are beneficial
    - Situations requiring a simple, parallelizable approach

    The `n_iter_restart` parameter controls the frequency of restarts. Shorter
    intervals lead to more exploration but less exploitation of each local region,
    while longer intervals allow more thorough local search before restarting.
    The optimal value depends on the problem's landscape and the expected basin
    of attraction size.

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
    epsilon : float
        The step-size for the climbing. Controls how far the optimizer looks
        for neighboring positions.
    distribution : str
        The type of distribution to sample neighbors from. Options are
        "normal", "laplace", "gumbel", or "logistic".
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    n_iter_restart : int
        Number of iterations between random restarts. After this many iterations,
        the optimizer jumps to a new random position. Default is 10.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RandomRestartHillClimbingOptimizer

    >>> def schwefel(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(418.9829 * 2 - x * np.sin(np.sqrt(abs(x)))
    ...              - y * np.sin(np.sqrt(abs(y))))

    >>> search_space = {
    ...     "x": np.linspace(-500, 500, 1000),
    ...     "y": np.linspace(-500, 500, 1000),
    ... }

    >>> opt = RandomRestartHillClimbingOptimizer(search_space, n_iter_restart=20)
    >>> opt.search(schwefel, n_iter=500)
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
        distribution: Literal["normal", "laplace", "gumbel", "logistic"] = "normal",
        n_neighbours: int = 3,
        n_iter_restart: int = 10,
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
            n_iter_restart=n_iter_restart,
        )
