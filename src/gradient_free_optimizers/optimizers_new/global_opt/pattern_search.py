# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Pattern Search (Hooke-Jeeves) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL

This optimizer uses the "State vor iterate()" pattern:
- Computes the full next position BEFORE _iterate_*_batch() methods are called
- The _iterate_*_batch() methods just extract the appropriate dimension components
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core_optimizer import CoreOptimizer

if TYPE_CHECKING:
    pass


def _arrays_equal(a, b):
    """Check if two arrays are element-wise equal."""
    if hasattr(a, "__len__") and hasattr(b, "__len__"):
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    return a == b


class PatternSearch(CoreOptimizer):
    """Pattern Search (Hooke-Jeeves) optimizer.

    Dimension Support:
        - Continuous: YES (coordinate-wise search)
        - Categorical: YES (random category switching in pattern)
        - Discrete: YES (coordinate-wise search with integer steps)

    Explores the search space by evaluating positions along coordinate axes
    from the current best position. The pattern size reduces when the best
    position is found within recent evaluations.

    The algorithm generates a symmetric pattern of 2*n_dims positions
    (positive and negative steps along each axis) and samples from them.

    This implementation uses the "State vor iterate()" pattern:
    The pattern state machine computes the full next position BEFORE
    the template methods are called. The _iterate_*_batch() methods
    simply extract the appropriate dimension components.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    n_positions : int, default=4
        Number of pattern positions to evaluate per iteration.
    pattern_size : float, default=0.25
        Initial pattern size as fraction of dimension size.
    reduction : float, default=0.9
        Factor to reduce pattern size when converging.
    """

    name = "Pattern Search"
    _name_ = "pattern_search"
    __name__ = "PatternSearch"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        n_positions: int = 4,
        pattern_size: float = 0.25,
        reduction: float = 0.9,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.n_positions = n_positions
        self.pattern_size = pattern_size
        self.reduction = reduction

        # Limit n_positions to number of dimensions
        n_dims = len(self.search_space)
        self.n_positions_ = min(n_positions, n_dims)
        self.pattern_size_tmp = pattern_size
        self.pattern_pos_l: list[np.ndarray] = []

        # Initialize RNG for reproducibility
        self._rng = np.random.default_rng(self.random_seed)

        # Pre-computed next position for "State vor iterate()" pattern
        self._next_position: np.ndarray | None = None
        self._next_position_computed: bool = False

    def _get_dim_sizes(self):
        """Get dimension sizes for pattern generation."""
        sizes = []
        for i, name in enumerate(self.search_space.keys()):
            dim_def = self.search_space[name]
            if isinstance(dim_def, tuple):
                # Continuous: range
                sizes.append(dim_def[1] - dim_def[0])
            elif isinstance(dim_def, list):
                # Categorical: number of categories
                sizes.append(len(dim_def))
            elif isinstance(dim_def, np.ndarray):
                # Discrete: length of array
                sizes.append(len(dim_def))
            else:
                sizes.append(1)
        return np.array(sizes)

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

    def generate_pattern(self, current_position: np.ndarray) -> None:
        """Generate pattern positions around the current position.

        Creates positions at +/- pattern_size * dim_size along each axis,
        then randomly samples n_positions_ from them.

        Parameters
        ----------
        current_position : np.ndarray
            Current position to generate pattern around.
        """
        pattern_pos_l = []

        n_valid_pos = len(self.positions_valid)
        n_pattern_pos = int(self.n_positions_ * 2)
        n_pos_min = min(n_valid_pos, n_pattern_pos)

        # Check if best is in recent positions (convergence detection)
        if n_pos_min > 0 and self.pos_best is not None:
            recent_positions = self.positions_valid[-n_pos_min:]
            best_in_recent_pos = any(
                _arrays_equal(np.array(self.pos_best), pos) for pos in recent_positions
            )
            if best_in_recent_pos:
                self.pattern_size_tmp *= self.reduction

        pattern_size = self.pattern_size_tmp
        dim_sizes = self._get_dim_sizes()
        dim_names = list(self.search_space.keys())

        for idx in range(len(dim_names)):
            pos_pattern_p = np.array(current_position, dtype=float)
            pos_pattern_n = np.array(current_position, dtype=float)

            dim_def = self.search_space[dim_names[idx]]

            if isinstance(dim_def, list):
                # Categorical: random switch instead of arithmetic
                n_cats = len(dim_def)
                if n_cats > 1:
                    # Pick two random different categories
                    current_cat = int(current_position[idx])
                    other_cats = [c for c in range(n_cats) if c != current_cat]
                    if other_cats:
                        pos_pattern_p[idx] = random.choice(other_cats)
                        pos_pattern_n[idx] = random.choice(other_cats)
            else:
                # Continuous or discrete: arithmetic step
                step = pattern_size * dim_sizes[idx]
                pos_pattern_p[idx] += step
                pos_pattern_n[idx] -= step

            pos_pattern_p = self._clip_to_bounds(pos_pattern_p)
            pos_pattern_n = self._clip_to_bounds(pos_pattern_n)

            pattern_pos_l.append(pos_pattern_p)
            pattern_pos_l.append(pos_pattern_n)

        # Sample n_positions_ from the pattern
        if len(pattern_pos_l) > self.n_positions_:
            self.pattern_pos_l = list(random.sample(pattern_pos_l, self.n_positions_))
        else:
            self.pattern_pos_l = pattern_pos_l

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

    def _finish_initialization(self) -> None:
        """Generate initial pattern around current position.

        Called by CoreOptimizer.finish_initialization() after all init
        positions have been evaluated. Sets up the first pattern for
        the iteration phase.

        Note: DO NOT set search_state here - CoreOptimizer handles that.
        """
        if self.pos_current is not None:
            self.generate_pattern(self.pos_current)

    def _compute_next_pattern_position(self) -> None:
        """Compute the full next position based on pattern state.

        This is called ONCE before the first _iterate_*_batch() method.
        The computed position is stored in self._next_position.

        Uses the "State vor iterate()" pattern.
        """
        if self._next_position_computed:
            return

        # Random restart check
        if random.random() < self.rand_rest_p:
            self._next_position = self._generate_random_position()
            self._next_position_computed = True
            return

        # Get next position from pattern
        if not self.pattern_pos_l:
            # Regenerate pattern if empty
            if self.pos_current is not None:
                self.generate_pattern(self.pos_current)
            else:
                self._next_position = self._generate_random_position()
                self._next_position_computed = True
                return

        # Pattern still empty after regeneration? Fall back to random
        if not self.pattern_pos_l:
            self._next_position = self._generate_random_position()
            self._next_position_computed = True
            return

        pos_new = self.pattern_pos_l.pop(0)

        # Handle constraints
        if not self.conv.not_in_constraint(pos_new):
            # Try perturbation to find valid position
            max_tries = 10
            for _ in range(max_tries):
                # Perturb position slightly
                noise = self._rng.normal(0, 0.1, len(pos_new))
                pos_new = self._clip_to_bounds(np.array(pos_new) + noise)
                if self.conv.not_in_constraint(pos_new):
                    break

            # Fallback to random if still invalid
            if not self.conv.not_in_constraint(pos_new):
                pos_new = self._generate_random_position()

        self._next_position = pos_new
        self._next_position_computed = True

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Extract continuous components from pre-computed pattern position.

        The pattern position is computed ONCE when this method is first called.
        """
        self._compute_next_pattern_position()
        return self._next_position[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Extract categorical components from pre-computed pattern position."""
        self._compute_next_pattern_position()
        return self._next_position[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Extract discrete components from pre-computed pattern position."""
        self._compute_next_pattern_position()
        return self._next_position[self._discrete_mask]

    def _evaluate(self, score_new: float) -> None:
        """Evaluate score and regenerate pattern when needed.

        After evaluating n_positions_ * 2 positions, regenerates the
        pattern and updates the current position to the best found.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Reset the pre-computed flag for next iteration
        self._next_position_computed = False

        # Check if we need to regenerate pattern
        modZero = self.nth_trial % int(self.n_positions_ * 2) == 0

        if modZero or len(self.pattern_pos_l) == 0:
            if self.search_state == "iter" and self.pos_current is not None:
                self.generate_pattern(self.pos_current)

            # Find best among recent positions
            if self.positions_valid and self.scores_valid:
                n_recent = min(self.n_positions_, len(self.scores_valid))
                score_new_list_temp = self.scores_valid[-n_recent:]
                pos_new_list_temp = self.positions_valid[-n_recent:]

                if score_new_list_temp:
                    idx = np.argmax(score_new_list_temp)
                    score = score_new_list_temp[idx]
                    pos = pos_new_list_temp[idx]

                    self._update_current(pos, score)
                    self._update_best(pos, score)
                    return

        # Standard tracking
        self._update_current(self.pos_new, score_new)
        self._update_best(self.pos_new, score_new)
