# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Parallel tempering optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
)


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, AskTell):
    """Parallel Tempering optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initial_evaluations : list[tuple[dict, float]]
        Previously evaluated parameters and their scores to seed the optimizer.
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
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 5,
        n_iter_swap: int = 5,
    ):
        if constraints is None:
            constraints = []
        if conditions is None:
            conditions = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            conditions=conditions,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            population=population,
            n_iter_swap=n_iter_swap,
        )

        self._process_initial_evaluations(initial_evaluations)
