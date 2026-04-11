# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""CMA-ES optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import (
    CMAESOptimizer as _CMAESOptimizer,
)


class CMAESOptimizer(_CMAESOptimizer, AskTell):
    """CMA-ES optimizer with ask/tell interface.

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
    population : int or None, default=None
        Number of candidate solutions sampled per generation.
    mu : int or None, default=None
        Number of best solutions selected as parents.
    sigma : float, default=0.3
        Initial step size as a fraction of the search space range.
    ipop_restart : bool, default=False
        Enable IPOP restart strategy with doubled population size.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = None,
        mu: int = None,
        sigma: float = 0.3,
        ipop_restart: bool = False,
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
            mu=mu,
            sigma=sigma,
            ipop_restart=ipop_restart,
        )

        self._process_initial_evaluations(initial_evaluations)
