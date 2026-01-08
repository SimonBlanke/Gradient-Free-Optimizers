# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from typing import Any, Callable

from ..base_optimizer import BaseOptimizer
from ..core_optimizer.converter import ArrayLike


def max_list_idx(list_: list[float]) -> int:
    max_item = max(list_)
    max_item_idx = [i for i, j in enumerate(list_) if j == max_item]
    return max_item_idx[-1:][0]


class HillClimbingOptimizer(BaseOptimizer):
    """Simple hill climbing optimizer that greedily moves toward better solutions.

    Evaluates multiple neighbors around the current position and moves to the
    best one. This is a local search algorithm that can get stuck in local optima.

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
        Probability of random restart to escape local optima.
    nth_process : int, optional
        Process index for parallel optimization.
    epsilon : float, default=0.03
        Step size for generating neighbors (fraction of search space).
    distribution : str, default="normal"
        Distribution for step sizes: "normal", "laplace", or "logistic".
    n_neighbours : int, default=3
        Number of neighbors to evaluate before selecting the best.
    """

    name = "Hill Climbing"
    _name_ = "hill_climbing"
    __name__ = "HillClimbingOptimizer"

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
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours

    @BaseOptimizer.track_new_pos
    @BaseOptimizer.random_iteration
    def iterate(self) -> ArrayLike:
        """Generate next position by climbing from current position."""
        return self.move_climb(
            self.pos_current,
            epsilon=self.epsilon,
            distribution=self.distribution,
        )

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new: float) -> None:
        """Evaluate score and update current position after n_neighbours trials."""
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:
            return

        modZero = self.nth_trial % self.n_neighbours == 0
        if modZero:
            score_new_list_temp = self.scores_valid[-self.n_neighbours :]
            pos_new_list_temp = self.positions_valid[-self.n_neighbours :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)
