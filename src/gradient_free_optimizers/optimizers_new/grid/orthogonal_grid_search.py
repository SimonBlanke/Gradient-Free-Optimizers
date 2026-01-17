# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Orthogonal Grid Search.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
Uses sequential nested loop traversal (dimension by dimension).
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class OrthogonalGridSearch(CoreOptimizer):
    """Orthogonal Grid Search - traverses dimensions sequentially.

    Dimension Support:
        - Continuous: LIMITED (must be discretized first)
        - Categorical: YES (enumerate all categories)
        - Discrete: YES (enumerate all values)

    Traverses the search space in a nested loop fashion, exhausting each
    dimension before moving to the next. This provides systematic coverage
    but may take longer to explore distant regions.

    The traversal is equivalent to:
        for x0 in range(dim_0_size):
            for x1 in range(dim_1_size):
                for x2 in range(dim_2_size):
                    ...
                    evaluate([x0, x1, x2, ...])

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
        Step multiplier for grid traversal (1 = visit every point).
    """

    name = "Orthogonal Grid Search"
    _name_ = "orthogonal_grid_search"
    __name__ = "OrthogonalGridSearch"

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

        # Use converter's dimension info (handles overflow via Python's arbitrary precision)
        self._dim_sizes = self.conv.dim_sizes
        self._search_space_size = self.conv.search_space_size

    def _grid_move(self):
        """Convert iteration number to a position in the multi-dimensional search space.

        Uses modular arithmetic to compute the position - essentially a
        mixed-radix number representation where each dimension has its own base.
        """
        # Convert to Python list to avoid numpy overflow
        dim_sizes = [int(s) for s in self._dim_sizes]

        # Account for step_size and handle wraparound for multiple passes
        mod_tmp = self.nth_trial * self.step_size + int(
            self.nth_trial * self.step_size / self._search_space_size
        )
        div_tmp = self.nth_trial * self.step_size + int(
            self.nth_trial * self.step_size / self._search_space_size
        )
        new_pos = []

        for dim_size in dim_sizes:
            mod = mod_tmp % dim_size
            div = int(div_tmp / dim_size)

            new_pos.append(mod)

            mod_tmp = div
            div_tmp = div

        return np.array(new_pos)

    def iterate(self):
        """Generate next orthogonal grid position.

        Returns
        -------
        np.ndarray
            The next grid position to evaluate.
        """
        pos_new = self._grid_move()
        pos_new = self._clip_position(pos_new)

        self.pos_new = pos_new
        self.pos_new_list.append(pos_new)

        return pos_new

    def _evaluate(self, score_new):
        """Simple greedy evaluation - just track best position."""
        self._update_best(self.pos_new, score_new)

        # Update current to new position (grid search always moves)
        self._update_current(self.pos_new, score_new)
