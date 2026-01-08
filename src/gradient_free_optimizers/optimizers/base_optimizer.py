# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from typing import Any, Callable

from .core_optimizer import CoreOptimizer


class BaseOptimizer(CoreOptimizer):
    """
    Base class for all gradient-free optimization algorithms.

    BaseOptimizer provides the foundation that all specific optimizer
    implementations inherit from. It handles common functionality like
    search space configuration, constraint handling, initialization
    strategies, and the basic optimization lifecycle.

    This class serves as the connection point between:

    - **CoreOptimizer**: Low-level position manipulation and tracking
    - **Specific Optimizers**: Algorithm-specific iteration logic
    - **Search**: High-level search orchestration

    Subclasses must implement:

    - ``iterate()``: Return the next position to evaluate based on
      the algorithm's strategy (e.g., hill climbing, evolutionary, Bayesian)
    - ``evaluate(score_new)``: Update internal state based on the score
      of the newly evaluated position

    Parameters
    ----------
    search_space : dict[str, array-like]
        Dictionary mapping parameter names to arrays of possible values.
        Each parameter is searched over its corresponding array.
    initialize : dict[str, int], default={"grid": 4, "random": 2, "vertices": 4}
        Initialization strategy. Keys can be:

        - "grid": Number of positions from a grid pattern
        - "random": Number of random positions
        - "vertices": Number of corner positions
        - "warm_start": List of dicts with explicit starting positions
    constraints : list[callable], optional
        List of constraint functions. Each function receives a parameter
        dict and returns True if the position is valid.
    random_state : int, optional
        Seed for random number generation for reproducibility.
    rand_rest_p : float, default=0
        Probability of performing a random exploration step instead of
        the algorithm's normal iteration. Useful for escaping local optima.
    nth_process : int, optional
        Process identifier for parallel optimization scenarios.

    Attributes
    ----------
    optimizers : list
        List containing this optimizer (for compatibility with ensemble methods).
    search_state : str
        Current state of the search: "init" or "iter".

    See Also
    --------
    CoreOptimizer : Provides position tracking and movement utilities.
    Search : Provides the high-level search interface.
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.optimizers = [self]

    def finish_initialization(self) -> None:
        """Transition from initialization phase to iteration phase."""
        self.search_state = "iter"

    def evaluate(self, score_new: float) -> None:
        """
        Update optimizer state based on a newly evaluated score.

        This base implementation only handles the initial case where
        no positions have been evaluated yet. Subclasses override this
        to implement algorithm-specific update logic (e.g., acceptance
        criteria for hill climbing, fitness updates for evolutionary).

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.pos_current = self.pos_new

            self.score_best = score_new
            self.score_current = score_new
