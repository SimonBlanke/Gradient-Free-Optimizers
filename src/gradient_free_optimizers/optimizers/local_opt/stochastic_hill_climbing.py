# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import math
from collections.abc import Callable
from random import random
from typing import Any

from ..core_optimizer.parameter_tracker.stochastic_hill_climbing import (
    ParameterTracker,
)
from . import HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HillClimbingOptimizer, ParameterTracker):
    """Hill climbing with probabilistic acceptance of worse solutions.

    Unlike standard hill climbing, this variant can accept worse solutions
    with a probability based on the score difference. This helps escape
    local optima while still preferring uphill moves.

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
    p_accept : float, default=0.5
        Base probability for accepting worse solutions.
    """

    name = "Stochastic Hill Climbing"
    _name_ = "stochastic_hill_climbing"
    __name__ = "StochasticHillClimbingOptimizer"

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
        p_accept: float = 0.5,
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

        self.p_accept = p_accept
        self.temp = 1

    @ParameterTracker.considered_transitions
    def _consider(self, p_accept: float) -> None:
        if p_accept >= random():
            self._execute_transition()

    @ParameterTracker.transitions
    def _execute_transition(self) -> None:
        self._new2current()

    @property
    def _normalized_energy_state(self) -> float:
        denom = self.score_current + self.score_new

        if denom == 0:
            return 1
        elif math.isinf(abs(denom)):
            return 0
        else:
            return (self.score_new - self.score_current) / denom

    @property
    def _exponent(self) -> float:
        if self.temp == 0:
            return -math.inf
        else:
            return self._normalized_energy_state / self.temp

    def _p_accept_default(self) -> float:
        try:
            exp_val = math.exp(self._exponent)
        except OverflowError:
            exp_val = math.inf
        return self.p_accept * 2 / (1 + exp_val)

    @HillClimbingOptimizer.track_new_score
    def _transition(self, score_new: float) -> None:
        p_accept = self._p_accept_default()
        self._consider(p_accept)

    def evaluate(self, score_new: float) -> None:
        if score_new <= self.score_current:
            self._transition(score_new)
        else:
            HillClimbingOptimizer.evaluate(self, score_new)
