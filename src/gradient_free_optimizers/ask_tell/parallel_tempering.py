# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Parallel tempering optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
)


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, AskTell):
    """Parallel Tempering optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        Constraint functions restricting the search space.
    random_state : int or None, default=None
        Seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart.
    population : int, default=5
        Number of parallel simulated annealers at different temperatures.
    n_iter_swap : int, default=5
        Number of iterations between temperature swap attempts.
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
            population=population,
            n_iter_swap=n_iter_swap,
        )
