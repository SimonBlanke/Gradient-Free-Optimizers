# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Grid search optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from ..optimizers import GridSearchOptimizer as _GridSearchOptimizer


class GridSearchOptimizer(_GridSearchOptimizer, AskTell):
    """Grid Search optimizer with ask/tell interface.

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
    step_size : int, default=1
        Step size between grid points in index space.
    direction : str, default="diagonal"
        Traversal pattern through the grid.
    resolution : int, default=100
        Number of grid points for continuous dimensions.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        step_size: int = 1,
        direction: Literal["diagonal", "orthogonal"] = "diagonal",
        resolution: int = 100,
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
            step_size=step_size,
            direction=direction,
            resolution=resolution,
        )

        self._process_initial_evaluations(initial_evaluations)
