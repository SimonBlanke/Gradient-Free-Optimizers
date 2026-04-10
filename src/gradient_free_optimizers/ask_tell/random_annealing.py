# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random annealing optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import RandomAnnealingOptimizer as _RandomAnnealingOptimizer


class RandomAnnealingOptimizer(_RandomAnnealingOptimizer, AskTell):
    """Random Annealing optimizer with ask/tell interface.

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
    epsilon : float, default=0.03
        Step size as a fraction of each dimension's range.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors to evaluate per iteration.
    annealing_rate : float, default=0.98
        Multiplicative cooling factor applied to temperature each iteration.
    start_temp : float, default=10
        Initial temperature that multiplies the base step size.
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
        epsilon: float = 0.03,
        distribution: Literal["normal", "laplace", "gumbel", "logistic"] = "normal",
        n_neighbours: int = 3,
        annealing_rate: float = 0.98,
        start_temp: float = 10,
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
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
            annealing_rate=annealing_rate,
            start_temp=start_temp,
        )
