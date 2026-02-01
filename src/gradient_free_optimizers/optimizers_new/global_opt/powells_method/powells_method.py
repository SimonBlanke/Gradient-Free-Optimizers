# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Powell's Conjugate Direction Method.

Supports: CONTINUOUS, DISCRETE_NUMERICAL
Note: Categorical dimensions have limited support.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ...core_optimizer import CoreOptimizer
from .direction import Direction
from .line_search import GoldenSectionLineSearch, GridLineSearch, HillClimbLineSearch

if TYPE_CHECKING:
    pass


class PowellsMethod(CoreOptimizer):
    """
    Powell's conjugate direction method for gradient-free optimization.

    Dimension Support:
        - Continuous: YES (line searches work naturally)
        - Categorical: LIMITED (line searches don't apply naturally)
        - Discrete: YES (line searches with rounding)

    This optimizer performs sequential line searches along a set of directions,
    updating the directions after each complete cycle to form conjugate directions.
    This leads to faster convergence than simple coordinate descent.

    Algorithm:
    1. Initialize with coordinate axis directions
    2. For each cycle:
       a. Save starting position
       b. Perform line search along each direction
       c. Compute displacement from cycle start to end
       d. Replace direction with largest improvement with the displacement direction
    3. Repeat until convergence

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
    epsilon : float, default=0.03
        Step size for hill_climb line search method.
    distribution : str, default="normal"
        Distribution for hill_climb line search perturbations.
    iters_p_dim : int, default=10
        Number of evaluations per direction during line search.
    line_search : str, default="grid"
        Line search method: "grid", "golden", or "hill_climb".
    convergence_threshold : float, default=1e-8
        Minimum total improvement per cycle to continue.
    """

    name = "Powell's Method"
    _name_ = "powells_method"
    __name__ = "PowellsMethod"

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
        epsilon: float = 0.03,
        distribution: str = "normal",
        iters_p_dim: int = 10,
        line_search: str = "grid",
        convergence_threshold: float = 1e-8,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        # Parameters for line search (used by HillClimbLineSearch)
        self.epsilon = epsilon
        self.distribution = distribution

        self.iters_p_dim = iters_p_dim
        self.line_search_method = line_search
        self.convergence_threshold = convergence_threshold

        if line_search not in ("grid", "golden", "hill_climb"):
            raise ValueError(
                f"line_search must be 'grid', 'golden', or 'hill_climb', "
                f"got '{line_search}'"
            )

        # State variables initialized in finish_initialization
        self.directions = None
        self.line_searcher = None
        self.current_direction_idx = 0
        self.cycle_start_pos = None
        self.direction_start_score = None
        self.direction_improvements = []
        self.converged = False

    def conv2pos_typed(self, pos):
        """Convert position to valid position with proper types."""
        pos_new = []
        dim_names = list(self.search_space.keys())

        for i, name in enumerate(dim_names):
            dim_def = self.search_space[name]
            val = pos[i]

            if isinstance(dim_def, tuple):
                # Continuous: clip to bounds
                pos_new.append(np.clip(val, dim_def[0], dim_def[1]))
            elif isinstance(dim_def, list):
                # Categorical: clip to valid indices
                pos_new.append(int(np.clip(round(val), 0, len(dim_def) - 1)))
            elif isinstance(dim_def, np.ndarray):
                # Discrete: clip to valid indices
                pos_new.append(int(np.clip(round(val), 0, len(dim_def) - 1)))
            else:
                pos_new.append(val)

        return np.array(pos_new)

    def _finish_initialization(self):
        """Set up the direction matrix and state after initialization phase."""
        n_dims = len(self.search_space)

        # Initialize directions as coordinate unit vectors
        self.directions = []
        for i in range(n_dims):
            self.directions.append(Direction.coordinate_axis(i, n_dims))

        # Create the line search strategy
        self.line_searcher = self._create_line_searcher()

        # State tracking
        self.current_direction_idx = 0
        self.cycle_start_pos = None
        self.direction_start_score = None
        self.direction_improvements = []
        self.converged = False

    def _create_line_searcher(self):
        """Create the appropriate line search strategy."""
        if self.line_search_method == "grid":
            return GridLineSearch(self)
        elif self.line_search_method == "golden":
            return GoldenSectionLineSearch(self)
        elif self.line_search_method == "hill_climb":
            return HillClimbLineSearch(self, self.epsilon, self.distribution)
        else:
            raise ValueError(f"Unknown line search method: {self.line_search_method}")

    def _start_direction_search(self):
        """Initialize line search for the current direction."""
        self.direction_start_score = self.score_current

        direction = self.directions[self.current_direction_idx]
        self.line_searcher.start(
            origin=self.pos_current.copy(),
            direction=direction.direction,
            max_iters=self.iters_p_dim,
        )

    def _finish_direction_search(self):
        """Complete line search and move to best position found."""
        best_pos, best_score = self.line_searcher.get_best_result()

        if best_score is not None:
            improvement = best_score - (self.direction_start_score or 0)
            self.direction_improvements.append(improvement)
            self._update_current(best_pos, best_score)
        else:
            self.direction_improvements.append(0.0)

        self.current_direction_idx += 1

    def _start_new_cycle(self):
        """Start a new cycle through all directions."""
        self.cycle_start_pos = (
            self.pos_current.copy() if self.pos_current is not None else None
        )
        self.current_direction_idx = 0
        self.direction_improvements = []

    def _complete_cycle(self):
        """Complete a cycle and update directions with conjugate direction."""
        if self.cycle_start_pos is None:
            return

        # Check for convergence based on total improvement this cycle
        if self.direction_improvements:
            total_improvement = sum(self.direction_improvements)
            if total_improvement < self.convergence_threshold:
                self.converged = True
                return

        displacement = np.array(self.pos_current) - np.array(self.cycle_start_pos)
        displacement_norm = np.linalg.norm(displacement)

        if displacement_norm > 1e-10 and self.direction_improvements:
            max_improve_idx = np.argmax(self.direction_improvements)
            try:
                new_direction = Direction(displacement)
                self.directions[max_improve_idx] = new_direction
            except ValueError:
                pass  # Displacement too small, keep old direction

    def _generate_position(self) -> np.ndarray:
        """Generate next position via Powell's line search.

        Powell's method performs line searches along conjugate directions.
        When converged or no valid position available, falls back to
        random position generation.

        Note: Constraint checking is handled by CoreOptimizer.iterate().

        Returns
        -------
        np.ndarray
            Next candidate position (may violate constraints).
        """
        # Random restart check
        if random.random() < self.rand_rest_p:
            return self.init.move_random_typed()

        # If converged, fall back to random exploration
        if self.converged:
            return self.init.move_random_typed()

        n_dims = len(self.search_space)

        # Handle state transitions
        while True:
            # Check if we completed all directions in current cycle
            if self.current_direction_idx >= n_dims:
                self._complete_cycle()
                # Check if we just converged
                if self.converged:
                    return self.init.move_random_typed()
                self._start_new_cycle()
                self._start_direction_search()
                break

            # Check if we need to start or continue a line search
            if not self.line_searcher.is_active():
                if self.cycle_start_pos is None:
                    # Very first iteration: start new cycle and first direction
                    self._start_new_cycle()
                    self._start_direction_search()
                    break
                elif self.direction_start_score is None:
                    # Direction not started yet: start it
                    self._start_direction_search()
                    break
                else:
                    # Direction search completed: finish and advance
                    self._finish_direction_search()
                    # Reset direction_start_score so next iteration starts new search
                    self.direction_start_score = None
                    # Loop will check if we need to start next direction or new cycle
                    continue
            else:
                # Line searcher is active, ready to get position
                break

        # Get next position from line searcher
        pos = self.line_searcher.get_next_position()

        # Fallback if no position available
        if pos is None:
            return self.init.move_random_typed()

        return pos

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Not used - Powell's method overrides _generate_position().

        Powell's method uses line searches along conjugate directions,
        not the standard batch iteration pattern.

        Raises
        ------
        NotImplementedError
            Always raised - see _generate_position() instead.
        """
        raise NotImplementedError(
            "Powell's method uses line search, not batch iteration. "
            "See _generate_position() for the algorithm implementation."
        )

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Not used - Powell's method overrides _generate_position()."""
        raise NotImplementedError(
            "Powell's method uses line search, not batch iteration. "
            "See _generate_position() for the algorithm implementation."
        )

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Not used - Powell's method overrides _generate_position()."""
        raise NotImplementedError(
            "Powell's method uses line search, not batch iteration. "
            "See _generate_position() for the algorithm implementation."
        )

    def _evaluate(self, score_new: float) -> None:
        """
        Evaluate a new score and update tracking.

        Parameters
        ----------
        score_new : float
            The score for the most recently evaluated position.

        Note: Unlike other optimizers, Powell's Method only updates the current
        position when a direction search completes (in _finish_direction_search).
        We only update the line searcher and global best here.
        """
        # Update line searcher with the result (only during iteration phase)
        if self.line_searcher is not None and self.pos_new is not None:
            self.line_searcher.update(self.pos_new, score_new)

        # Only update global best tracking, NOT current position.
        # Current position updated in _finish_direction_search when line search done.
        self._update_best(self.pos_new, score_new)
