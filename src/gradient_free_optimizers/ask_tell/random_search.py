# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random search optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import RandomSearchOptimizer as _RandomSearchOptimizer


class RandomSearchOptimizer(_RandomSearchOptimizer, AskTell):
    """Random Search optimizer with ask/tell interface.

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
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        random_state: int = None,
    ):
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            random_state=random_state,
        )

        self._process_initial_evaluations(initial_evaluations)
