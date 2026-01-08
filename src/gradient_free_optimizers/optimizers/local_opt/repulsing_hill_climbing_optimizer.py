# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core_optimizer.converter import ArrayLike
from . import HillClimbingOptimizer


class RepulsingHillClimbingOptimizer(HillClimbingOptimizer):
    """Hill climbing with repulsion from worse solutions.

    When a worse solution is found, the step size is multiplied by a repulsion
    factor to escape the current region faster. This helps avoid getting stuck
    in flat or declining areas of the search space.

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
    epsilon : float, default=0.03
        Base step size for generating neighbors.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors to evaluate.
    repulsion_factor : float, default=5
        Multiplier for step size when escaping worse regions.
    """

    name = "Repulsing Hill Climbing"
    _name_ = "repulsing_hill_climbing"
    __name__ = "RepulsingHillClimbingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        epsilon: float = 0.03,
        distribution: str = "normal",
        n_neighbours: int = 3,
        repulsion_factor: float = 5,
    ) -> None:
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
        )

        self.tabus = []
        self.repulsion_factor = repulsion_factor
        self.epsilon_mod = 1

    @HillClimbingOptimizer.track_new_pos
    def iterate(self) -> ArrayLike:
        """Generate next position with adaptive step size."""
        return self.move_climb(
            self.pos_current,
            epsilon=self.epsilon,
            distribution=self.distribution,
            epsilon_mod=self.epsilon_mod,
        )

    def evaluate(self, score_new: float) -> None:
        super().evaluate(score_new)

        if score_new <= self.score_current:
            self.epsilon_mod = self.repulsion_factor
        else:
            self.epsilon_mod = 1
