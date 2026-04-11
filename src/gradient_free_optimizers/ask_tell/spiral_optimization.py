# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Spiral optimization with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import SpiralOptimization as _SpiralOptimization


class SpiralOptimization(_SpiralOptimization, AskTell):
    """Spiral Optimization with ask/tell interface.

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
    population : int, default=10
        Number of search agents in the spiral population.
    decay_rate : float, default=0.99
        Controls how quickly the spiral radius contracts.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 10,
        decay_rate: float = 0.99,
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
            decay_rate=decay_rate,
        )

        self._process_initial_evaluations(initial_evaluations)
