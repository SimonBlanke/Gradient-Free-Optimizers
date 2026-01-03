# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Optional, Tuple, List

from gradient_free_optimizers._array_backend import array, maximum, argmax, random as np_random

from .base import LineSearch


class HillClimbLineSearch(LineSearch):
    """
    Hill climbing line search strategy.

    Performs stochastic hill climbing constrained to the search direction.
    Uses random perturbations along the direction to explore and exploit.
    Good for escaping local optima but may be less precise than golden section.
    """

    def __init__(self, optimizer, epsilon: float = 0.03, distribution: str = "normal"):
        """
        Initialize hill climb line search.

        Parameters
        ----------
        optimizer : PowellsMethod
            Reference to parent optimizer.
        epsilon : float
            Step size scaling factor for perturbations.
        distribution : str
            Distribution for random steps (currently uses normal distribution).
        """
        super().__init__(optimizer)
        self.epsilon = epsilon
        self.distribution = distribution

        self.origin = None
        self.direction = None
        self.max_iters: int = 0
        self.current_step: int = 0
        self.active: bool = False

        # Track current best position for hill climbing
        self.current_pos = None
        self.current_score: Optional[float] = None

        # Track all evaluations
        self.evaluated_positions: List = []
        self.evaluated_scores: List[float] = []

    def start(
        self,
        origin,
        direction,
        max_iters: int,
    ) -> None:
        """Initialize hill climb search along the direction."""
        self.origin = array(origin).copy()
        self.direction = array(direction).copy()
        self.max_iters = max_iters
        self.current_step = 0
        self.active = True

        self.current_pos = array(origin).copy()
        self.current_score = None

        self.evaluated_positions = []
        self.evaluated_scores = []

    def get_next_position(self):
        """Generate next position by perturbing along the direction."""
        if not self.active or self.current_step >= self.max_iters:
            self.active = False
            return None

        # Generate random step along direction
        max_positions = self.optimizer.conv.max_positions
        sigma = self.epsilon * float(maximum(max_positions, 0))
        t = np_random.normal(0, sigma)

        pos_float = self.current_pos + t * self.direction
        pos = self._snap_to_grid(pos_float)

        # If position violates constraints, try again with smaller step
        attempts = 0
        while not self._is_valid(pos) and attempts < 10:
            t *= 0.5
            pos_float = self.current_pos + t * self.direction
            pos = self._snap_to_grid(pos_float)
            attempts += 1

        if not self._is_valid(pos):
            # Fallback: return current position
            pos = self.current_pos.copy()

        self.current_step += 1
        return pos

    def update(self, position, score: float) -> None:
        """Update current position if new score is better."""
        self.evaluated_positions.append(array(position).copy())
        self.evaluated_scores.append(score)

        # Hill climbing: move to new position if it's better
        if self.current_score is None or score > self.current_score:
            self.current_pos = array(position).copy()
            self.current_score = score

    def get_best_result(self) -> Tuple:
        """Return the best position found during hill climbing."""
        if not self.evaluated_scores:
            return None, None

        best_idx = argmax(self.evaluated_scores)
        return self.evaluated_positions[best_idx], self.evaluated_scores[best_idx]

    def is_active(self) -> bool:
        """Check if more hill climb steps should be taken."""
        return self.active and self.current_step < self.max_iters
