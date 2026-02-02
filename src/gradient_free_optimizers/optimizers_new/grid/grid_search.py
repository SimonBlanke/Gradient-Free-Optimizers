# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Grid Search Optimizer.

Supports: DISCRETE_NUMERICAL, CATEGORICAL, CONTINUOUS (auto-discretized)

Continuous dimensions are automatically discretized using the `resolution`
parameter, which specifies how many grid points to create.

Template Method Pattern Compliance:
    - Does NOT override iterate() - keeps public interface intact
    - Overrides _generate_position() for grid-specific position generation
    - Constraint handling via iterate()'s retry loop naturally advances grid
"""

from functools import reduce
from math import gcd

import numpy as np

from ..core_optimizer import CoreOptimizer


def _discretize_search_space(search_space, resolution):
    """Convert continuous dimensions to discrete grids.

    Parameters
    ----------
    search_space : dict
        Original search space with potential continuous dimensions (tuples).
    resolution : int
        Number of grid points for continuous dimensions.

    Returns
    -------
    dict
        Search space with continuous dimensions converted to numpy arrays.
    """
    discretized = {}
    for name, space in search_space.items():
        if isinstance(space, tuple) and len(space) == 2:
            # Continuous dimension: convert to linspace
            low, high = space
            if isinstance(low, int | float) and isinstance(high, int | float):
                discretized[name] = np.linspace(low, high, resolution)
            else:
                discretized[name] = space
        else:
            discretized[name] = space
    return discretized


class GridSearchOptimizer(CoreOptimizer):
    """Systematic grid search over the entire search space.

    Dimension Support:
        - Continuous: YES (auto-discretized using `resolution` parameter)
        - Categorical: YES (enumerate all categories)
        - Discrete: YES (enumerate all values)

    Evaluates positions in a structured grid pattern, either diagonally
    (visiting diverse regions early) or orthogonally (dimension by dimension).

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
        Continuous dimensions (tuples) are automatically discretized.
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
    resolution : int, default=100
        Number of grid points for continuous dimensions. Higher values
        give finer resolution but require more iterations to cover.
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
        resolution=100,
    ):
        # Auto-discretize continuous dimensions before parent init
        self.resolution = resolution
        discretized_space = _discretize_search_space(search_space, resolution)

        super().__init__(
            search_space=discretized_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.step_size = step_size
        self.direction = direction

        if direction not in ("diagonal", "orthogonal"):
            raise ValueError(
                f"direction must be 'diagonal' or 'orthogonal', got '{direction}'"
            )

        # Grid state
        self._grid_counter = 0  # Tracks position in linearized space
        self._diagonal_direction = None  # Prime generator for diagonal (lazy init)

        # Use converter's dimension info (handles overflow via arbitrary precision)
        self._dim_sizes = self.conv.dim_sizes
        self._search_space_size = self.conv.search_space_size

    # =========================================================================
    # Diagonal Traversal
    # =========================================================================

    def _get_diagonal_direction(self):
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

    def _compute_diagonal_position(self):
        """Convert current grid counter to position using diagonal traversal.

        Uses a bijection from Z/(search_space_size * Z) to the product space.
        """
        # Convert to Python list to avoid numpy overflow
        dim_sizes = [int(s) for s in self._dim_sizes]

        # Compute 1D pointer from grid counter
        # Multiple passes support via step_size
        current_pass = self._grid_counter % self.step_size
        base_pointer = self._grid_counter // self.step_size

        # Apply prime-based diagonal step
        high_dim_pointer = (
            current_pass + base_pointer * self._diagonal_direction
        ) % self._search_space_size

        # Convert 1D pointer to N-D coordinates
        new_pos = []
        pointer = high_dim_pointer

        for dim in range(len(dim_sizes) - 1):
            remaining_prod = reduce(lambda x, y: x * y, dim_sizes[dim + 1 :], 1)
            new_pos.append(pointer // remaining_prod % dim_sizes[dim])
            pointer = pointer % remaining_prod
        new_pos.append(pointer)

        return np.array(new_pos)

    # =========================================================================
    # Orthogonal Traversal
    # =========================================================================

    def _compute_orthogonal_position(self):
        """Convert current grid counter to position using orthogonal traversal.

        Uses modular arithmetic for mixed-radix decomposition.
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

    # =========================================================================
    # Template Method Override
    # =========================================================================

    def _generate_position(self):
        """Generate next grid position based on configured direction.

        This method overrides CoreOptimizer._generate_position() to provide
        grid-specific position generation. It:
        1. Computes the grid position for current counter (diagonal or orthogonal)
        2. Advances the counter (supporting constraint retry)
        3. Clips to valid bounds

        Returns
        -------
        np.ndarray
            Clipped position array ready for evaluation.
        """
        if self.direction == "diagonal":
            # Lazy initialization of diagonal direction
            if self._diagonal_direction is None:
                self._diagonal_direction = self._get_diagonal_direction()
            pos = self._compute_diagonal_position()
        else:  # orthogonal
            pos = self._compute_orthogonal_position()

        # Advance counter for next call (supports constraint retry)
        self._grid_counter += 1

        # Clip to valid bounds (handles edge cases)
        return self._clip_position(pos)

    # =========================================================================
    # Template Method Stubs (not used - _generate_position bypasses them)
    # =========================================================================

    def _iterate_continuous_batch(self) -> "np.ndarray":
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "GridSearch uses _generate_position() for grid traversal"
        )

    def _iterate_categorical_batch(self) -> "np.ndarray":
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "GridSearch uses _generate_position() for grid traversal"
        )

    def _iterate_discrete_batch(self) -> "np.ndarray":
        """Not used - grid search generates full position via _generate_position."""
        raise NotImplementedError(
            "GridSearch uses _generate_position() for grid traversal"
        )

    def _evaluate(self, score_new):
        """Track best position using greedy evaluation.

        Grid search always moves to the next position (deterministic traversal),
        so acceptance is always True. We just track the best found so far.
        """
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)
