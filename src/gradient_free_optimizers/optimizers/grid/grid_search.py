# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from ..base_optimizer import BaseOptimizer
from ..core_optimizer.converter import ArrayLike
from .diagonal_grid_search import DiagonalGridSearchOptimizer
from .orthogonal_grid_search import OrthogonalGridSearchOptimizer


class GridSearchOptimizer(BaseOptimizer):
    """Systematic grid search over the entire search space.

    Evaluates positions in a structured grid pattern, either diagonally
    (visiting diverse regions early) or orthogonally (dimension by dimension).

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    step_size : int, default=1
        Step size for grid traversal (1 = visit every point).
    direction : str, default="diagonal"
        Grid traversal direction: "diagonal" or "orthogonal".
    """

    name = "Grid Search"
    _name_ = "grid_search"
    __name__ = "GridSearchOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        step_size: int = 1,
        direction: Literal["diagonal", "orthogonal"] = "diagonal",
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.step_size = step_size
        self.direction = direction

        if direction == "orthogonal":
            self.grid_search_opt = OrthogonalGridSearchOptimizer(
                search_space=search_space,
                initialize=initialize,
                constraints=constraints,
                random_state=random_state,
                rand_rest_p=rand_rest_p,
                nth_process=nth_process,
                step_size=step_size,
            )
        elif direction == "diagonal":
            self.grid_search_opt = DiagonalGridSearchOptimizer(
                search_space=search_space,
                initialize=initialize,
                constraints=constraints,
                random_state=random_state,
                rand_rest_p=rand_rest_p,
                nth_process=nth_process,
                step_size=step_size,
            )
        else:
            raise ValueError(
                f"direction must be 'diagonal' or 'orthogonal', got '{direction}'"
            )

    @BaseOptimizer.track_new_pos
    def iterate(self) -> ArrayLike:
        return self.grid_search_opt.iterate()

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new: float) -> None:
        self.grid_search_opt.evaluate(score_new)
