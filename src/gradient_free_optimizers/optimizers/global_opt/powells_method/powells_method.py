# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._array_backend import array, argmax, linalg

from ...local_opt import HillClimbingOptimizer
from .direction import Direction
from .line_search import GridLineSearch, GoldenSectionLineSearch, HillClimbLineSearch


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
        iters_p_dim=10,
        line_search="grid",
        convergence_threshold=1e-8,
    ):
        """
        Initialize Powell's Method optimizer.

        Parameters
        ----------
        search_space : dict
            Dictionary defining the search space for each parameter.
        initialize : dict, optional
            Initialization strategy.
        constraints : list, optional
            List of constraint functions.
        random_state : int, optional
            Random seed for reproducibility.
        rand_rest_p : float, optional
            Probability of random restart.
        nth_process : int, optional
            Process number for parallel execution.
        epsilon : float, optional
            Step size for hill climbing line search.
        distribution : str, optional
            Distribution for hill climbing perturbations.
        n_neighbours : int, optional
            Number of neighbors for hill climbing.
        iters_p_dim : int, optional
            Number of evaluations per direction during line search.
        line_search : str, optional
            Line search method: "grid", "golden", or "hill_climb".
        convergence_threshold : float, optional
            Minimum total improvement per cycle to continue. If the sum of
            improvements across all directions falls below this threshold,
            the optimizer switches to random exploration.
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

        self.iters_p_dim = iters_p_dim
        self.line_search_method = line_search
        self.epsilon = epsilon
        self.distribution = distribution
        self.convergence_threshold = convergence_threshold

        if line_search not in ("grid", "golden", "hill_climb"):
            raise ValueError(
                f"line_search must be 'grid', 'golden', or 'hill_climb', "
                f"got '{line_search}'"
            )

    def finish_initialization(self):
        """Set up the direction matrix and state after initialization phase."""
        n_dims = self.conv.n_dimensions

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
            improvement = best_score - self.direction_start_score
            self.direction_improvements.append(improvement)
            self._eval2current(best_pos, best_score)
        else:
            self.direction_improvements.append(0.0)

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

        # Check for convergence based on total improvement this cycle
        if self.direction_improvements:
            total_improvement = sum(self.direction_improvements)
            if total_improvement < self.convergence_threshold:
                self.converged = True
                return

        displacement = array(self.pos_current) - array(self.cycle_start_pos)
        displacement_norm = linalg.norm(displacement)

        if displacement_norm > 1e-10 and self.direction_improvements:
            max_improve_idx = argmax(self.direction_improvements)
            try:
                new_direction = Direction(displacement)
                self.directions[max_improve_idx] = new_direction
            except ValueError:
                pass  # Displacement too small, keep old direction

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        """Generate the next position to evaluate."""
        # If converged, fall back to hill climbing exploration
        if self.converged:
            return self.move_climb(
                self.pos_current,
                epsilon=self.epsilon,
                distribution=self.distribution,
            )

        n_dims = self.conv.n_dimensions

        # Handle state transitions
        while True:
            # Check if we completed all directions in current cycle
            if self.current_direction_idx >= n_dims:
                self._complete_cycle()
                # Check if we just converged
                if self.converged:
                    return self.move_climb(
                        self.pos_current,
                        epsilon=self.epsilon,
                        distribution=self.distribution,
                    )
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
        pos_new = self.line_searcher.get_next_position()

        # Fallback if no position available
        if pos_new is None:
            pos_new = self.move_climb(
                self.pos_current,
                epsilon=self.epsilon,
                distribution=self.distribution,
            )

        return pos_new

    @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new):
        """
        Evaluate a new score and update tracking.

        Parameters
        ----------
        score_new : float
            The score for the most recently evaluated position.
        """
        # Update line searcher with the result (only during iteration phase)
        if hasattr(self, "line_searcher") and self.pos_new is not None:
            self.line_searcher.update(self.pos_new, score_new)

        # Call parent evaluate
        super(HillClimbingOptimizer, self).evaluate(score_new)
