# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Pattern search optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import PatternSearch as _PatternSearch


class PatternSearch(_PatternSearch, AskTell):
    """Pattern Search optimizer with ask/tell interface.

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
    n_positions : int, default=4
        Number of positions in the search pattern.
    pattern_size : float, default=0.25
        Initial pattern size as a fraction of each dimension's range.
    reduction : float, default=0.9
        Factor by which the pattern size is reduced when no improvement is found.
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
        n_positions: int = 4,
        pattern_size: float = 0.25,
        reduction: float = 0.9,
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
            n_positions=n_positions,
            pattern_size=pattern_size,
            reduction=reduction,
        )
