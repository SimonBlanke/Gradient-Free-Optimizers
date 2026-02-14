# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Diagonal Grid Search.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
Uses number theory (prime generators) to traverse the search space diagonally.

Template Method Pattern Compliance:
    - Does NOT override iterate() - keeps public interface intact
    - Overrides _generate_position() for grid-specific position generation
    - Constraint handling via iterate()'s retry loop naturally advances grid
"""

from functools import reduce
from math import gcd

import numpy as np

from ..base_optimizer import BaseOptimizer


class DiagonalGridSearch(BaseOptimizer):
    """Diagonal Grid Search - traverses grid using prime generators.

    Dimension Support:
        - Continuous: LIMITED (must be discretized first)
        - Categorical: YES (enumerate all categories)
        - Discrete: YES (enumerate all values)

    Uses a prime generator to traverse the search space diagonally,
    ensuring diverse coverage early in the search. The traversal visits
    positions that are spread across multiple dimensions simultaneously.

    This is mathematically equivalent to finding a generator of the cyclic
    group Z/(n*Z) where n is the total search space size. The prime-based
    approach ensures we visit all points exactly once before repeating.

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

    name = "Diagonal Grid Search"
    _name_ = "diagonal_grid_search"
    __name__ = "DiagonalGridSearch"

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
        self._grid_counter = 0  # Tracks position in 1D linearized space
        self._direction = None  # Prime generator (computed lazily)

        # Use converter's dimension info (handles overflow via arbitrary precision)
        self._dim_sizes = self.conv.dim_sizes
        self._search_space_size = self.conv.search_space_size

    def _get_direction(self):
        """Generate a prime number to serve as direction in search space.

        As direction is prime with the search space size, we know it is
        a generator of Z/(search_space_size*Z).
        """
        n_dims = len(self.search_space)
        search_space_size = self._search_space_size

        # Find prime number near search_space_size ** (1/n_dims)
        dim_root = int(round(search_space_size ** (1 / n_dims)))
        is_prime = False

        while not is_prime:
            if gcd(int(search_space_size), int(dim_root)) == 1:
                is_prime = True
            else:
                dim_root -= 1

        return dim_root

    def _compute_grid_position(self):
        """Convert current grid counter to a position in the multi-dimensional space.

        Uses a bijection from Z/(search_space_size * Z) to the product space.
        The grid counter is updated BEFORE calling this method to support
        constraint retry (each retry gets the next position).

        Returns
        -------
        np.ndarray
            Position array with indices for each dimension.
        """
        # Convert to Python list to avoid numpy overflow
        dim_sizes = [int(s) for s in self._dim_sizes]

        # Compute 1D pointer from grid counter
        # Multiple passes support via step_size
        current_pass = self._grid_counter % self.step_size
        base_pointer = self._grid_counter // self.step_size

        # Apply prime-based diagonal step
        high_dim_pointer = (
            current_pass + base_pointer * self._direction
        ) % self._search_space_size

        # Convert 1D pointer to N-D coordinates
        # Bijection: Z/search_space_size*Z -> (Z/dim_1*Z)x...x(Z/dim_n*Z)
        new_pos = []
        pointer = high_dim_pointer

        for dim in range(len(dim_sizes) - 1):
            # Use Python's native multiplication to avoid overflow
            remaining_prod = reduce(lambda x, y: x * y, dim_sizes[dim + 1 :], 1)
            new_pos.append(pointer // remaining_prod % dim_sizes[dim])
            pointer = pointer % remaining_prod
        new_pos.append(pointer)

        return np.array(new_pos)

    def _generate_position(self):
        """Generate next grid position using diagonal traversal.

        This method overrides CoreOptimizer._generate_position() to provide
        grid-specific position generation. It:
        1. Lazily initializes the prime direction on first call
        2. Computes the grid position for current counter
        3. Advances the counter (supporting constraint retry)
        4. Clips to valid bounds

        Returns
        -------
        np.ndarray
            Clipped position array ready for evaluation.
        """
        # Lazy initialization of direction
        if self._direction is None:
            self._direction = self._get_direction()

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
            "DiagonalGridSearch uses _generate_position() for grid traversal"
        )

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "DiagonalGridSearch uses _generate_position() for grid traversal"
        )

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "DiagonalGridSearch uses _generate_position() for grid traversal"
        )

    def _on_evaluate(self, score_new):
        """Track best position using greedy evaluation.

        Grid search always moves to the next position (deterministic traversal),
        so acceptance is always True. We just track the best found so far.
        """
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)
