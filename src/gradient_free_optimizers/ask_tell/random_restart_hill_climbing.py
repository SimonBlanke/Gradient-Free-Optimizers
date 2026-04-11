# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random restart hill climbing optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from ..optimizers import (
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
)


class RandomRestartHillClimbingOptimizer(_RandomRestartHillClimbingOptimizer, AskTell):
    """Random Restart Hill Climbing optimizer with ask/tell interface.

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
    epsilon : float, default=0.03
        Step size as a fraction of each dimension's range.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors to evaluate per iteration.
    n_iter_restart : int, default=10
        Number of iterations between random restarts.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        epsilon: float = 0.03,
        distribution: Literal["normal", "laplace", "gumbel", "logistic"] = "normal",
        n_neighbours: int = 3,
        n_iter_restart: int = 10,
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
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
            n_iter_restart=n_iter_restart,
        )

        self._process_initial_evaluations(initial_evaluations)
