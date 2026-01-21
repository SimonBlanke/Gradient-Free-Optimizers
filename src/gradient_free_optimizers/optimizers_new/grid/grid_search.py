# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Grid Search Optimizer.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
Note: Continuous dimensions must be discretized for grid search.
"""

import numpy as np

from ..core_optimizer import CoreOptimizer
from .diagonal_grid_search import DiagonalGridSearch
from .orthogonal_grid_search import OrthogonalGridSearch


class GridSearchOptimizer(CoreOptimizer):
    """Systematic grid search over the entire search space.

    Dimension Support:
        - Continuous: LIMITED (must be discretized first)
        - Categorical: YES (enumerate all categories)
        - Discrete: YES (enumerate all values)

    Evaluates positions in a structured grid pattern, either diagonally
    (visiting diverse regions early) or orthogonally (dimension by dimension).

    The main GridSearchOptimizer is a facade that delegates to either
    DiagonalGridSearch or OrthogonalGridSearch based on the direction parameter.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    step_size : int, default=1
        Step size for grid traversal (1 = visit every point).
    direction : str, default="diagonal"
        Grid traversal direction: "diagonal" or "orthogonal".
    """

    name = "Grid Search"
    _name_ = "grid_search"
    __name__ = "GridSearchOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        step_size=1,
        direction="diagonal",
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.step_size = step_size
        self.direction = direction

        # Create the appropriate grid search implementation
        if direction == "orthogonal":
            self.grid_search_opt = OrthogonalGridSearch(
                search_space=search_space,
                initialize=initialize,
                constraints=constraints,
                random_state=random_state,
                rand_rest_p=rand_rest_p,
                nth_process=nth_process,
                step_size=step_size,
            )
        elif direction == "diagonal":
            self.grid_search_opt = DiagonalGridSearch(
                search_space=search_space,
                initialize=initialize,
                constraints=constraints,
                random_state=random_state,
                rand_rest_p=rand_rest_p,
                nth_process=nth_process,
                step_size=step_size,
            )
        else:
            raise ValueError(
                f"direction must be 'diagonal' or 'orthogonal', got '{direction}'"
            )

    def iterate(self):
        """Delegate iteration to the underlying grid search implementation.

        Returns
        -------
        np.ndarray
            The next grid position to evaluate.
        """
        pos = self.grid_search_opt.iterate()

        # Track in parent optimizer (property setter auto-appends to pos_new_list)
        self.pos_new = pos

        return pos

    def _iterate_continuous_batch(self) -> "np.ndarray":
        """Not used by GridSearch - uses systematic grid traversal."""
        raise NotImplementedError("GridSearch uses systematic grid traversal")

    def _iterate_categorical_batch(self) -> "np.ndarray":
        """Not used by GridSearch - uses systematic grid traversal."""
        raise NotImplementedError("GridSearch uses systematic grid traversal")

    def _iterate_discrete_batch(self) -> "np.ndarray":
        """Not used by GridSearch - uses systematic grid traversal."""
        raise NotImplementedError("GridSearch uses systematic grid traversal")

    def _evaluate(self, score_new):
        """Greedy evaluation - track best position.

        Grid search doesn't need complex acceptance criteria - it just
        systematically walks through the grid. We track the best position
        found so far.
        """
        self._update_best(self.pos_new, score_new)

        # Update current to new position (grid search always moves)
        self._update_current(self.pos_new, score_new)
