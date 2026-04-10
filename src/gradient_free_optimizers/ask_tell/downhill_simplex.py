# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Downhill simplex optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import DownhillSimplexOptimizer as _DownhillSimplexOptimizer


class DownhillSimplexOptimizer(_DownhillSimplexOptimizer, AskTell):
    """Downhill Simplex optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initialize : dict, optional
        Strategy for generating initial positions.
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
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        alpha: float = 1,
        gamma: float = 2,
        beta: float = 0.5,
        sigma: float = 0.5,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
        )
