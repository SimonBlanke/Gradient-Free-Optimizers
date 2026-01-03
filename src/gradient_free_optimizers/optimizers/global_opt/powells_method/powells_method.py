# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ...local_opt import HillClimbingOptimizer
from .direction import Direction


# Golden ratio for golden section search
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2


class PowellsMethod(HillClimbingOptimizer):
    """
    Powell's conjugate direction method for gradient-free optimization.

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
    """

    name = "Powell's Method"
    _name_ = "powells_method"
    __name__ = "PowellsMethod"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        iters_per_direction=10,
        line_search="grid",
    ):
        """
        Initialize Powell's Method optimizer.

        Parameters
        ----------
        search_space : dict
            Dictionary defining the search space for each parameter
        initialize : dict, optional
            Initialization strategy
        constraints : list, optional
            List of constraint functions
        random_state : int, optional
            Random seed for reproducibility
        rand_rest_p : float, optional
            Probability of random restart
        nth_process : int, optional
            Process number for parallel execution
        epsilon : float, optional
            Step size for hill climbing line search
        distribution : str, optional
            Distribution for hill climbing perturbations
        n_neighbours : int, optional
            Number of neighbors for hill climbing
        iters_per_direction : int, optional
            Number of evaluations per direction during line search
        line_search : str, optional
            Line search method: "grid", "golden", or "hill_climb"
        """
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
        )

        self.iters_per_direction = iters_per_direction
        self.line_search_method = line_search

        if line_search not in ("grid", "golden", "hill_climb"):
            raise ValueError(
                f"line_search must be 'grid', 'golden', or 'hill_climb', got '{line_search}'"
            )

    def finish_initialization(self):
        """Set up the direction matrix and state after initialization phase."""
        n_dims = self.conv.n_dimensions

        # Initialize directions as coordinate unit vectors
        self.directions = []
        for i in range(n_dims):
            self.directions.append(Direction.coordinate_axis(i, n_dims))

        # State tracking
        self.current_direction_idx = 0
        self.cycle_start_pos = None
        self.direction_start_pos = None
        self.direction_start_score = None
        self.direction_improvements = []

        # Line search state
        self.line_search_step = 0
        self.direction_search_active = False  # Flag to track if search is in progress
        self.grid_positions = []  # Pre-generated positions for grid search
        self.evaluated_positions = []  # Positions that have been evaluated
        self.evaluated_scores = []  # Corresponding scores

        # For golden section search
        self.golden_state = None

    def _compute_max_step(self, origin, direction):
        """
        Compute the maximum step size along a direction that stays within bounds.

        Parameters
        ----------
        origin : np.ndarray
            Starting position
        direction : np.ndarray
            Normalized direction vector

        Returns
        -------
        float
            Maximum step size (positive direction)
        """
        max_t_positive = float("inf")
        max_t_negative = float("inf")

        for i, (d, o, max_pos) in enumerate(
            zip(direction, origin, self.conv.max_positions)
        ):
            if abs(d) < 1e-10:
                continue

            if d > 0:
                # Steps to reach max boundary
                t_to_max = (max_pos - o) / d
                # Steps to reach 0 boundary (negative direction)
                t_to_zero = -o / d
                max_t_positive = min(max_t_positive, t_to_max)
                max_t_negative = min(max_t_negative, -t_to_zero)
            else:
                # d < 0
                t_to_zero = -o / d
                t_to_max = (max_pos - o) / d
                max_t_positive = min(max_t_positive, t_to_zero)
                max_t_negative = min(max_t_negative, -t_to_max)

        # Use the smaller of the two to get symmetric bounds
        max_t = min(max_t_positive, max_t_negative)

        # Ensure we have at least some range to search
        if max_t < 1:
            max_t = max(self.conv.max_positions) * 0.5

        return max_t

    def _get_grid_positions(self, origin, direction, n_steps):
        """
        Generate grid positions along a direction for line search.

        Parameters
        ----------
        origin : np.ndarray
            Starting position
        direction : Direction
            Search direction
        n_steps : int
            Number of positions to generate

        Returns
        -------
        list of np.ndarray
            Valid positions along the line
        """
        max_t = self._compute_max_step(origin, direction.direction)

        positions = []
        t_values = np.linspace(-max_t, max_t, n_steps)

        for t in t_values:
            pos_float = direction.get_position_at(origin, t)
            pos_valid = self.conv2pos(pos_float)

            # Avoid duplicates
            is_duplicate = any(np.array_equal(pos_valid, p) for p in positions)
            if not is_duplicate and self.conv.not_in_constraint(pos_valid):
                positions.append(pos_valid)

        return positions

    def _start_direction_search(self):
        """Initialize state for searching along the current direction."""
        self.direction_start_pos = self.pos_current.copy()
        self.direction_start_score = self.score_current
        self.line_search_step = 0
        self.direction_search_active = True  # Flag to track if search is in progress

        # Separate lists: positions to evaluate vs evaluated results
        self.grid_positions = []  # Pre-generated positions for grid search
        self.evaluated_positions = []  # Positions that have been evaluated
        self.evaluated_scores = []  # Corresponding scores

        if self.line_search_method == "grid":
            # Pre-generate all positions for grid search
            direction = self.directions[self.current_direction_idx]
            self.grid_positions = self._get_grid_positions(
                self.direction_start_pos, direction, self.iters_per_direction
            )
        elif self.line_search_method == "golden":
            # Initialize golden section state
            # Standard golden section: a < c < d < b
            # c = a + (1-phi)(b-a), d = a + phi(b-a) where phi ≈ 0.618
            direction = self.directions[self.current_direction_idx]
            max_t = self._compute_max_step(
                self.direction_start_pos, direction.direction
            )
            a = -max_t
            b = max_t
            # Use 1-GOLDEN_RATIO for c so that c < d
            c = a + (1 - GOLDEN_RATIO) * (b - a)  # ≈ a + 0.382*(b-a)
            d = a + GOLDEN_RATIO * (b - a)        # ≈ a + 0.618*(b-a)
            self.golden_state = {
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "fc": None,
                "fd": None,
                "phase": "eval_c",
            }

    def _finish_direction_search(self):
        """Complete the search along current direction and move to best position."""
        self.direction_search_active = False  # Mark search as complete

        if self.evaluated_scores:
            best_idx = np.argmax(self.evaluated_scores)
            best_score = self.evaluated_scores[best_idx]
            best_pos = self.evaluated_positions[best_idx]

            # Calculate improvement for this direction
            improvement = best_score - self.direction_start_score
            self.direction_improvements.append(improvement)

            # Update current position to best found
            self._eval2current(best_pos, best_score)
        else:
            # No valid positions found, no improvement
            self.direction_improvements.append(0.0)

        # Move to next direction
        self.current_direction_idx += 1

    def _start_new_cycle(self):
        """Start a new cycle through all directions."""
        self.cycle_start_pos = self.pos_current.copy()
        self.current_direction_idx = 0
        self.direction_improvements = []

    def _complete_cycle(self):
        """Complete a cycle and update directions with conjugate direction."""
        if self.cycle_start_pos is None:
            return

        # Compute displacement from cycle start to end
        displacement = self.pos_current - self.cycle_start_pos
        displacement_norm = np.linalg.norm(displacement)

        if displacement_norm > 1e-10 and self.direction_improvements:
            # Find direction with largest improvement
            max_improve_idx = np.argmax(self.direction_improvements)

            # Replace that direction with the new conjugate direction
            try:
                new_direction = Direction(displacement)
                self.directions[max_improve_idx] = new_direction
            except ValueError:
                # Displacement was too small, keep old direction
                pass

    def _iterate_grid(self):
        """Generate next position for grid-based line search."""
        if self.line_search_step < len(self.grid_positions):
            pos = self.grid_positions[self.line_search_step]
            self.line_search_step += 1
            return pos
        else:
            # Exhausted grid positions, finish this direction
            return None

    def _iterate_golden(self):
        """Generate next position for golden section line search."""
        direction = self.directions[self.current_direction_idx]
        state = self.golden_state

        if state["phase"] == "eval_c":
            pos_float = direction.get_position_at(self.direction_start_pos, state["c"])
            pos = self.conv2pos(pos_float)
            return pos
        elif state["phase"] == "eval_d":
            pos_float = direction.get_position_at(self.direction_start_pos, state["d"])
            pos = self.conv2pos(pos_float)
            return pos
        else:
            return None

    def _update_golden_state(self, score):
        """Update golden section search state after evaluation."""
        state = self.golden_state

        if state["phase"] == "eval_c":
            state["fc"] = score
            if state["fd"] is None:
                state["phase"] = "eval_d"
            else:
                self._golden_narrow_bracket()
        elif state["phase"] == "eval_d":
            state["fd"] = score
            if state["fc"] is None:
                state["phase"] = "eval_c"
            else:
                self._golden_narrow_bracket()

    def _golden_narrow_bracket(self):
        """Narrow the bracket in golden section search.

        With c < d (standard golden section layout):
        - If f(c) > f(d): maximum in [a, d], narrow right side
        - If f(d) >= f(c): maximum in [c, b], narrow left side
        """
        state = self.golden_state

        if state["fc"] > state["fd"]:
            # Maximum is in [a, d], narrow from right
            # New bracket: [a, d], reuse c as new d
            state["b"] = state["d"]
            state["d"] = state["c"]
            state["fd"] = state["fc"]
            # Calculate new c: c = a + (1-phi)(b-a)
            state["c"] = state["a"] + (1 - GOLDEN_RATIO) * (state["b"] - state["a"])
            state["fc"] = None
            state["phase"] = "eval_c"
        else:
            # Maximum is in [c, b], narrow from left
            # New bracket: [c, b], reuse d as new c
            state["a"] = state["c"]
            state["c"] = state["d"]
            state["fc"] = state["fd"]
            # Calculate new d: d = a + phi(b-a)
            state["d"] = state["a"] + GOLDEN_RATIO * (state["b"] - state["a"])
            state["fd"] = None
            state["phase"] = "eval_d"

        self.line_search_step += 1

    def _iterate_hill_climb(self):
        """Generate next position for hill climbing along direction."""
        direction = self.directions[self.current_direction_idx]

        # Perturb along the direction only
        t = np.random.normal(0, self.epsilon * np.max(self.conv.max_positions))
        pos_float = direction.get_position_at(self.pos_current, t)
        pos = self.conv2pos(pos_float)

        if self.conv.not_in_constraint(pos):
            self.line_search_step += 1
            return pos

        # Fallback to standard move_climb if constrained
        return self.move_climb(
            self.pos_current, epsilon=self.epsilon, distribution=self.distribution
        )

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        """Generate the next position to evaluate."""
        n_dims = self.conv.n_dimensions

        # Check if we need to start a new cycle
        if self.current_direction_idx >= n_dims:
            self._complete_cycle()
            self._start_new_cycle()
            self._start_direction_search()

        # Check if we need to start searching a new direction
        if not self.direction_search_active:
            if self.cycle_start_pos is None:
                self._start_new_cycle()
            self._start_direction_search()

        # Generate next position based on line search method
        pos_new = None

        if self.line_search_method == "grid":
            pos_new = self._iterate_grid()
        elif self.line_search_method == "golden":
            if self.line_search_step < self.iters_per_direction:
                pos_new = self._iterate_golden()
        elif self.line_search_method == "hill_climb":
            if self.line_search_step < self.iters_per_direction:
                pos_new = self._iterate_hill_climb()

        # If line search for current direction is complete, move to next
        if pos_new is None:
            self._finish_direction_search()

            # Start next direction if available
            if self.current_direction_idx < n_dims:
                self._start_direction_search()

                if self.line_search_method == "grid":
                    pos_new = self._iterate_grid()
                elif self.line_search_method == "golden":
                    pos_new = self._iterate_golden()
                elif self.line_search_method == "hill_climb":
                    pos_new = self._iterate_hill_climb()

        # Fallback if still no position
        if pos_new is None:
            pos_new = self.move_climb(
                self.pos_current, epsilon=self.epsilon, distribution=self.distribution
            )

        return pos_new

    @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new):
        """
        Evaluate a new score and update tracking.

        Parameters
        ----------
        score_new : float
            The score for the most recently evaluated position
        """
        # Track for line search
        if self.pos_new is not None:
            self.evaluated_positions.append(self.pos_new.copy())
            self.evaluated_scores.append(score_new)

        # Update golden section state if applicable
        if self.line_search_method == "golden" and self.golden_state is not None:
            self._update_golden_state(score_new)

        # Call parent evaluate
        super(HillClimbingOptimizer, self).evaluate(score_new)
