# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random search optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import RandomSearchOptimizer as _RandomSearchOptimizer


class RandomSearchOptimizer(_RandomSearchOptimizer, AskTell):
    """Random Search optimizer with ask/tell interface.

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
        )
