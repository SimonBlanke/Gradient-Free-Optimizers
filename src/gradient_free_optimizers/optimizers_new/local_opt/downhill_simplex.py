# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Downhill Simplex (Nelder-Mead) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL

This optimizer uses the "State vor iterate()" pattern:
- Computes the full next position BEFORE _iterate_*_batch() methods are called
- The _iterate_*_batch() methods just extract the appropriate dimension components
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..base_optimizer import BaseOptimizer

if TYPE_CHECKING:
    pass


def _arrays_equal(a: Any, b: Any) -> bool:
    """Check if two arrays are element-wise equal."""
    if hasattr(a, "__len__") and hasattr(b, "__len__"):
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    return a == b


def _sort_list_idx(list_: list[float]) -> list[int]:
    """Return indices that would sort the list in descending order."""
    indexed = list(enumerate(list_))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in indexed]


def _centroid(array_list: list) -> np.ndarray:
    """Calculate centroid of a list of arrays."""
    n_dims = len(array_list[0])
    result = []

    for idx in range(n_dims):
        center_dim_pos = [arr[idx] for arr in array_list]
        center_dim_mean = sum(center_dim_pos) / len(center_dim_pos)
        result.append(center_dim_mean)

    return np.array(result)


class DownhillSimplexOptimizer(BaseOptimizer):
    """Nelder-Mead downhill simplex optimizer.

    Dimension Support:
        - Continuous: YES (native simplex operations)
        - Categorical: YES (with index-based operations)
        - Discrete: YES (with index-based operations)

    The Nelder-Mead simplex method maintains a simplex of n+1 points in
    n-dimensional space and iteratively transforms it through reflection,
    expansion, contraction, and shrinkage operations to find the optimum.

    This implementation uses the "State vor iterate()" pattern:
    The simplex state machine computes the full next position BEFORE
    the template methods are called. The _iterate_*_batch() methods
    simply extract the appropriate dimension components.

    Algorithm:
    1. Initialize simplex with n+1 vertices
    2. Sort vertices by function value
    3. Compute centroid of best n vertices
    4. Try reflection: if better than best, try expansion
    5. If reflection is middle-quality, accept it
    6. If reflection is poor, try contraction
    7. If contraction fails, shrink entire simplex toward best
    8. Repeat until convergence

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial simplex vertices.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    alpha : float, default=1.0
        Reflection coefficient.
    gamma : float, default=2.0
        Expansion coefficient.
    beta : float, default=0.5
        Contraction coefficient.
    sigma : float, default=0.5
        Shrinkage coefficient.
    """

    name = "Downhill Simplex"
    _name_ = "downhill_simplex"
    __name__ = "DownhillSimplexOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        alpha: float = 1.0,
        gamma: float = 2.0,
        beta: float = 0.5,
        sigma: float = 0.5,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.alpha = alpha  # Reflection coefficient
        self.gamma = gamma  # Expansion coefficient
        self.beta = beta  # Contraction coefficient
        self.sigma = sigma  # Shrink coefficient

        # Number of simplex vertices = dimensions + 1
        self.n_simp_positions = len(self.search_space) + 1

        # Simplex state
        self.simplex_pos = []
        self.simplex_scores = []
        self.simplex_step = 0

        # Ensure we have enough initial positions for the simplex
        diff_init = self.n_simp_positions - self.init.n_inits
        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        # Working variables for simplex operations
        self.center_array = None
        self.r_pos = None
        self.r_score = None
        self.e_pos = None
        self.e_score = None
        self.c_pos = None
        self.c_score = None
        self.h_pos = None
        self.h_score = None
        self.compress_idx = 0

        # Pre-computed next position for "State vor iterate()" pattern
        self._next_position = None
        self._next_position_computed = False

        # RNG for random fallback
        self._rng = np.random.default_rng(self.random_seed)

    def _clip_to_bounds(self, pos: np.ndarray) -> np.ndarray:
        """Clip position to valid bounds with proper types.

        Handles continuous, categorical, and discrete dimensions.
        """
        pos_clipped = pos.copy()
        dim_names = list(self.search_space.keys())

        for i, name in enumerate(dim_names):
            dim_def = self.search_space[name]
            val = pos[i]

            if isinstance(dim_def, tuple):
                # Continuous: clip to bounds
                pos_clipped[i] = np.clip(val, dim_def[0], dim_def[1])
            elif isinstance(dim_def, list):
                # Categorical: clip to valid indices
                pos_clipped[i] = int(np.clip(round(val), 0, len(dim_def) - 1))
            elif isinstance(dim_def, np.ndarray):
                # Discrete: clip to valid indices
                pos_clipped[i] = int(np.clip(round(val), 0, len(dim_def) - 1))

        return pos_clipped

    def _finish_initialization(self) -> None:
        """Initialize the simplex from evaluated positions.

        This hook is called by CoreOptimizer.finish_initialization() after
        all init positions have been evaluated. We use the evaluated positions
        to form the initial simplex vertices.
        """
        # Sort initial positions by score (best first)
        idx_sorted = _sort_list_idx(self.scores_valid)
        self.simplex_pos = [self.positions_valid[idx].copy() for idx in idx_sorted]
        self.simplex_scores = [self.scores_valid[idx] for idx in idx_sorted]

        self.simplex_step = 1

    def _compute_next_simplex_position(self) -> None:
        """Compute the full next position based on simplex state machine.

        This is called ONCE before the first _iterate_*_batch() method.
        The computed position is stored in self._next_position.
        """
        if self._next_position_computed:
            return

        # Check if simplex is stale (all vertices same)
        simplex_stale = all(
            _arrays_equal(self.simplex_pos[0], arr) for arr in self.simplex_pos
        )

        if simplex_stale:
            # Reset simplex from all evaluated positions
            idx_sorted = _sort_list_idx(self.scores_valid)
            self.simplex_pos = [
                self.positions_valid[idx].copy()
                for idx in idx_sorted[: self.n_simp_positions]
            ]
            self.simplex_scores = [
                self.scores_valid[idx] for idx in idx_sorted[: self.n_simp_positions]
            ]
            self.simplex_step = 1

        if self.simplex_step == 1:
            # Step 1: Reflection
            # Sort simplex by score (best first)
            idx_sorted = _sort_list_idx(self.simplex_scores)
            self.simplex_pos = [self.simplex_pos[idx] for idx in idx_sorted]
            self.simplex_scores = [self.simplex_scores[idx] for idx in idx_sorted]

            # Compute centroid of all but worst vertex
            self.center_array = _centroid(self.simplex_pos[:-1])

            # Reflection point: center + alpha * (center - worst)
            r_pos = self.center_array + self.alpha * (
                self.center_array - self.simplex_pos[-1]
            )
            self.r_pos = self._clip_to_bounds(r_pos)
            self._next_position = self.r_pos.copy()

        elif self.simplex_step == 2:
            # Step 2: Expansion
            e_pos = self.center_array + self.gamma * (
                self.center_array - self.simplex_pos[-1]
            )
            self.e_pos = self._clip_to_bounds(e_pos)
            self._next_position = self.e_pos.copy()

        elif self.simplex_step == 3:
            # Step 3: Contraction
            c_pos = self.h_pos + self.beta * (self.center_array - self.h_pos)
            c_pos = self._clip_to_bounds(c_pos)
            self._next_position = c_pos.copy()

        elif self.simplex_step == 4:
            # Step 4: Shrink
            pos = self.simplex_pos[self.compress_idx]
            pos = np.array(pos) + self.sigma * (
                np.array(self.simplex_pos[0]) - np.array(pos)
            )
            self._next_position = self._clip_to_bounds(pos)

        else:
            # Fallback: random position
            self._next_position = self._generate_random_position()

        self._next_position_computed = True

    def _generate_random_position(self) -> np.ndarray:
        """Generate a random position within bounds."""
        n_dims = len(self.search_space)
        pos = np.empty(n_dims)
        dim_names = list(self.search_space.keys())

        for i, name in enumerate(dim_names):
            dim_def = self.search_space[name]

            if isinstance(dim_def, tuple):
                # Continuous
                pos[i] = self._rng.uniform(dim_def[0], dim_def[1])
            elif isinstance(dim_def, list):
                # Categorical
                pos[i] = self._rng.integers(0, len(dim_def))
            elif isinstance(dim_def, np.ndarray):
                # Discrete
                pos[i] = self._rng.integers(0, len(dim_def))

        return pos

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Extract continuous components from pre-computed simplex position.

        The simplex position is computed ONCE when this method is first called.
        """
        self._compute_next_simplex_position()
        return self._next_position[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Extract categorical components from pre-computed simplex position."""
        self._compute_next_simplex_position()
        return self._next_position[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Extract discrete components from pre-computed simplex position."""
        self._compute_next_simplex_position()
        return self._next_position[self._discrete_mask]

    def _evaluate(self, score_new: float) -> None:
        """Evaluate score and update simplex state machine.

        Parameters
        ----------
        score_new : float
            Score for the most recently evaluated position.
        """
        # Reset the pre-computed flag for next iteration
        self._next_position_computed = False

        prev_pos = self._pos_new

        if self.simplex_step == 1:
            # Evaluate reflection
            self.r_score = score_new

            if self.r_score > self.simplex_scores[0]:
                # Reflection is better than best: try expansion
                self.simplex_step = 2

            elif self.r_score > self.simplex_scores[-2]:
                # Reflection is better than second-worst: accept it
                self.simplex_pos[-1] = self.r_pos.copy()
                self.simplex_scores[-1] = self.r_score
                self.simplex_step = 1

            else:
                # Reflection is poor: prepare for contraction
                if self.simplex_scores[-1] > self.r_score:
                    self.h_pos = np.array(self.simplex_pos[-1])
                    self.h_score = self.simplex_scores[-1]
                else:
                    self.h_pos = np.array(self.r_pos)
                    self.h_score = self.r_score

                self.simplex_step = 3

        elif self.simplex_step == 2:
            # Evaluate expansion
            self.e_score = score_new

            if self.e_score > self.r_score:
                # Expansion is better: use it
                self.simplex_pos[-1] = self.e_pos.copy()
                self.simplex_scores[-1] = self.e_score
            else:
                # Reflection was better: use reflection
                self.simplex_pos[-1] = self.r_pos.copy()
                self.simplex_scores[-1] = self.r_score

            self.simplex_step = 1

        elif self.simplex_step == 3:
            # Evaluate contraction
            self.c_pos = prev_pos
            self.c_score = score_new

            if self.c_score > self.simplex_scores[-1]:
                # Contraction improved: accept it
                self.simplex_scores[-1] = self.c_score
                self.simplex_pos[-1] = self.c_pos.copy()
                self.simplex_step = 1
            else:
                # Contraction failed: start shrink
                self.simplex_step = 4
                self.compress_idx = 1  # Start from second vertex (skip best)

        elif self.simplex_step == 4:
            # Evaluate shrink
            self.simplex_scores[self.compress_idx] = score_new
            self.simplex_pos[self.compress_idx] = prev_pos.copy()

            self.compress_idx += 1

            if self.compress_idx >= self.n_simp_positions:
                # Shrink complete: back to reflection
                self.simplex_step = 1

        # Update best position tracking
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)
