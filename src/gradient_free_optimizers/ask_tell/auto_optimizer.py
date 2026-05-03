# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""AutoOptimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import AutoOptimizer as _AutoOptimizer
from ..optimizers.pop_opt._selection_strategy import SelectionStrategy


class AutoOptimizer(_AutoOptimizer, AskTell):
    """Automatic optimizer with ask/tell interface.

    Maintains a heterogeneous portfolio of optimization algorithms and
    adaptively allocates iterations using a time-weighted selection
    strategy. This variant provides batch-capable ask/tell methods
    instead of the managed search() loop.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initial_evaluations : list[tuple[dict, float]]
        Previously evaluated parameters and their scores to seed the
        optimizer. Must contain at least as many entries as there are
        optimizers in the portfolio (3 by default).
    constraints : list, optional
        Constraint functions restricting the search space.
    random_state : int or None, default=None
        Seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart.
    portfolio : list of optimizer classes, optional
        Optimizer types in the portfolio. Defaults to RandomSearch,
        HillClimbing, RepulsingHillClimbing.
    strategy : SelectionStrategy, optional
        Strategy controlling optimizer selection. Defaults to
        DefaultStrategy with time-weighted UCB1.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        portfolio: list = None,
        strategy: SelectionStrategy = None,
    ):
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            portfolio=portfolio,
            strategy=strategy,
        )

        self._process_initial_evaluations(initial_evaluations)
