# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Diagonal Grid Search.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
Uses number theory (prime generators) to traverse the search space diagonally.
"""

from functools import reduce
from math import gcd

import numpy as np

from ..core_optimizer import CoreOptimizer


class DiagonalGridSearch(CoreOptimizer):
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

        # Initialize grid state
        self.initial_position = np.zeros(len(search_space), dtype=int)
        self.high_dim_pointer = 0  # Current position in 1D space
        self.direction_calc = None  # Prime generator (computed on first iterate)

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

    def _grid_move(self):
        """Convert 1D pointer to a position in the multi-dimensional search space.

        Uses a bijection from Z/(search_space_size * Z) to the product space.
        """
        new_pos = []
        # Convert to Python list to avoid numpy overflow
        dim_sizes = [int(s) for s in self._dim_sizes]
        pointer = self.high_dim_pointer

        # The coordinate of our new position for each dimension is
        # the quotient of the pointer by the product of remaining dimensions.
        # Bijection: Z/search_space_size*Z -> (Z/dim_1*Z)x...x(Z/dim_n*Z)
        for dim in range(len(dim_sizes) - 1):
            # Use Python's native multiplication to avoid overflow
            remaining_prod = reduce(lambda x, y: x * y, dim_sizes[dim + 1 :], 1)
            new_pos.append(pointer // remaining_prod % dim_sizes[dim])
            pointer = pointer % remaining_prod
        new_pos.append(pointer)

        return np.array(new_pos)

    def iterate(self):
        """Generate next diagonal grid position.

        Uses the prime-based diagonal traversal to generate the next position
        to evaluate.

        Returns
        -------
        np.ndarray
            The next grid position to evaluate.
        """
        while True:
            # If this is the first iteration:
            # Generate the direction and return initial_position
            if self.direction_calc is None:
                self.direction_calc = self._get_direction()

                pos_new = self.initial_position.copy()
                if self.conv.not_in_constraint(pos_new):
                    self.pos_new = pos_new
                    self.pos_new_list.append(pos_new)
                    return pos_new
                else:
                    # If initial position violates constraints, use random
                    pos_new = self._move_random()
                    self.pos_new = pos_new
                    self.pos_new_list.append(pos_new)
                    return pos_new

            # If this is not the first iteration:
            # Update high_dim_pointer by taking a step of size step_size * direction.

            # Multiple passes are needed in order to observe the entire search space
            # depending on the step_size parameter.
            current_pass = self.high_dim_pointer % self.step_size
            current_pass_finished = (
                (self.nth_trial + 1) * self.step_size // self._search_space_size
                > self.nth_trial * self.step_size // self._search_space_size
            )

            # Begin the next pass if current is finished.
            if current_pass_finished:
                self.high_dim_pointer = current_pass + 1
            else:
                # Otherwise update pointer in Z/(search_space_size*Z)
                # using the prime step direction and step_size.
                self.high_dim_pointer = (
                    self.high_dim_pointer + self.step_size * self.direction_calc
                ) % self._search_space_size

            # Compute corresponding position in our search space.
            pos_new = self._grid_move()
            pos_new = self._clip_position(pos_new)

            if self.conv.not_in_constraint(pos_new):
                self.pos_new = pos_new
                self.pos_new_list.append(pos_new)
                return pos_new

    def _move_random(self):
        """Generate a random valid position."""
        while True:
            pos = np.array([self._rng.integers(0, size) for size in self._dim_sizes])
            if self.conv.not_in_constraint(pos):
                return pos

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Not used - uses systematic diagonal grid traversal."""
        raise NotImplementedError("DiagonalGridSearch uses systematic traversal")

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Not used - uses systematic diagonal grid traversal."""
        raise NotImplementedError("DiagonalGridSearch uses systematic traversal")

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Not used - uses systematic diagonal grid traversal."""
        raise NotImplementedError("DiagonalGridSearch uses systematic traversal")

    def _evaluate(self, score_new):
        """Track best position using greedy evaluation."""
        self._update_best(self.pos_new, score_new)

        # Update current to new position (grid search always moves)
        self._update_current(self.pos_new, score_new)
