# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Downhill simplex optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import DownhillSimplexOptimizer as _DownhillSimplexOptimizer


class DownhillSimplexOptimizer(_DownhillSimplexOptimizer, AskTell):
    """Downhill Simplex optimizer with ask/tell interface.

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
    alpha : float, default=1
        Reflection coefficient.
    gamma : float, default=2
        Expansion coefficient.
    beta : float, default=0.5
        Contraction coefficient.
    sigma : float, default=0.5
        Shrink coefficient.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        alpha: float = 1,
        gamma: float = 2,
        beta: float = 0.5,
        sigma: float = 0.5,
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
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
        )

        self._process_initial_evaluations(initial_evaluations)
