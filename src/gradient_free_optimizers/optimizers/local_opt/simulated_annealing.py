# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..local_opt import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    """Simulated annealing optimizer inspired by metallurgical annealing.

    Uses a temperature parameter that decreases over time, controlling the
    probability of accepting worse solutions. High temperature allows more
    exploration; low temperature focuses on exploitation.

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
        Step size for generating neighbors.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors to evaluate.
    annealing_rate : float, default=0.97
        Temperature decay rate per iteration (temp *= annealing_rate).
    start_temp : float, default=1
        Initial temperature value.
    """

    name = "Simulated Annealing"
    _name_ = "simulated_annealing"
    __name__ = "SimulatedAnnealingOptimizer"

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
        annealing_rate: float = 0.97,
        start_temp: float = 1,
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

        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp

    def _p_accept_default(self):
        # the 'minus' is omitted, because we maximize a score
        try:
            return math.exp(self._exponent)
        except OverflowError:
            return math.inf

    def evaluate(self, score_new):
        StochasticHillClimbingOptimizer.evaluate(self, score_new)
        self.temp *= self.annealing_rate
