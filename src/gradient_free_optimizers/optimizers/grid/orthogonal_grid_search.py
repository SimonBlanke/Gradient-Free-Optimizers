# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Orthogonal Grid Search.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
Uses sequential nested loop traversal (dimension by dimension).

Template Method Pattern Compliance:
    - Does NOT override iterate() - keeps public interface intact
    - Overrides _generate_position() for grid-specific position generation
    - Constraint handling via iterate()'s retry loop naturally advances grid
"""

import numpy as np

from ..base_optimizer import BaseOptimizer


class OrthogonalGridSearch(BaseOptimizer):
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

        # Grid state
        self._grid_counter = 0  # Tracks position in linearized space

        # Use converter's dimension info (handles overflow via arbitrary precision)
        self._dim_sizes = self.conv.dim_sizes
        self._search_space_size = self.conv.search_space_size

    def _compute_grid_position(self):
        """Convert current grid counter to a position in the multi-dimensional space.

        Uses modular arithmetic to compute the position - essentially a
        mixed-radix number representation where each dimension has its own base.

        Returns
        -------
        np.ndarray
            Position array with indices for each dimension.
        """
        # Convert to Python list to avoid numpy overflow
        dim_sizes = [int(s) for s in self._dim_sizes]

        # Account for step_size and handle wraparound for multiple passes
        effective_counter = self._grid_counter * self.step_size
        wraparound_offset = effective_counter // self._search_space_size
        linear_index = effective_counter + wraparound_offset

        # Convert linear index to N-D coordinates using mixed-radix decomposition
        new_pos = []
        remainder = linear_index

        for dim_size in dim_sizes:
            new_pos.append(remainder % dim_size)
            remainder = remainder // dim_size

        return np.array(new_pos)

    def _generate_position(self):
        """Generate next grid position using orthogonal traversal.

        This method overrides CoreOptimizer._generate_position() to provide
        grid-specific position generation. It:
        1. Computes the grid position for current counter
        2. Advances the counter (supporting constraint retry)
        3. Clips to valid bounds

        Returns
        -------
        np.ndarray
            Clipped position array ready for evaluation.
        """
        # Compute position for current counter
        pos = self._compute_grid_position()

        # Advance counter for next call (supports constraint retry)
        self._grid_counter += 1

        # Clip to valid bounds (handles edge cases)
        return self._clip_position(pos)

    # =========================================================================
    # Template Method Stubs (not used - _generate_position bypasses them)
    # =========================================================================
    # These methods are required by the interface but are not called because
    # _generate_position() is overridden. Grid search generates the full
    # position at once, not by dimension type.

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "OrthogonalGridSearch uses _generate_position() for grid traversal"
        )

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "OrthogonalGridSearch uses _generate_position() for grid traversal"
        )

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "OrthogonalGridSearch uses _generate_position() for grid traversal"
        )

    def _on_evaluate(self, score_new):
        """Track best position using greedy evaluation.

        Grid search always moves to the next position (deterministic traversal),
        so acceptance is always True. We just track the best found so far.
        """
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)
