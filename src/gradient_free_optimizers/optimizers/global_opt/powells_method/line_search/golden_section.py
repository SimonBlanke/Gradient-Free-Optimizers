# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Optional, Tuple, List

import numpy as np

from .base import LineSearch


# Golden ratio: (sqrt(5) - 1) / 2 ≈ 0.618
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2


class GoldenSectionLineSearch(LineSearch):
    """
    Golden section search strategy for line search.

    Uses the golden ratio to efficiently narrow down the bracket containing
    the optimum. More efficient than grid search for unimodal functions,
    requiring O(log(n)) evaluations to achieve precision n.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.origin: Optional[np.ndarray] = None
        self.direction: Optional[np.ndarray] = None
        self.max_iters: int = 0
        self.current_step: int = 0
        self.active: bool = False

        # Bracket state: a < c < d < b
        self.a: float = 0.0
        self.b: float = 0.0
        self.c: float = 0.0
        self.d: float = 0.0
        self.fc: Optional[float] = None
        self.fd: Optional[float] = None
        self.phase: str = "eval_c"

        # Track all evaluations for best result
        self.evaluated_positions: List[np.ndarray] = []
        self.evaluated_scores: List[float] = []

    def start(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_iters: int,
    ) -> None:
        """Initialize golden section search with bracket [a, b]."""
        self.origin = origin.copy()
        self.direction = direction.copy()
        self.max_iters = max_iters
        self.current_step = 0
        self.active = True

        self.evaluated_positions = []
        self.evaluated_scores = []

        # Compute bracket bounds
        max_t = self._compute_max_step(origin, direction)

        # Initialize bracket: a < c < d < b
        self.a = -max_t
        self.b = max_t
        # c at ~38.2% of interval, d at ~61.8% of interval
        self.c = self.a + (1 - GOLDEN_RATIO) * (self.b - self.a)
        self.d = self.a + GOLDEN_RATIO * (self.b - self.a)

        self.fc = None
        self.fd = None
        self.phase = "eval_c"

    def get_next_position(self) -> Optional[np.ndarray]:
        """Return the next position to evaluate based on current phase."""
        if not self.active or self.current_step >= self.max_iters:
            self.active = False
            return None

        if self.phase == "eval_c":
            t = self.c
        elif self.phase == "eval_d":
            t = self.d
        else:
            self.active = False
            return None

        pos_float = self.origin + t * self.direction
        pos = self._snap_to_grid(pos_float)
        return pos

    def update(self, position: np.ndarray, score: float) -> None:
        """Update bracket based on evaluation result."""
        self.evaluated_positions.append(position.copy())
        self.evaluated_scores.append(score)

        if self.phase == "eval_c":
            self.fc = score
            if self.fd is None:
                # First time: need to evaluate d next
                self.phase = "eval_d"
            else:
                # Both c and d evaluated: narrow bracket
                self._narrow_bracket()
        elif self.phase == "eval_d":
            self.fd = score
            if self.fc is None:
                # Need to evaluate c next
                self.phase = "eval_c"
            else:
                # Both c and d evaluated: narrow bracket
                self._narrow_bracket()

    def _narrow_bracket(self) -> None:
        """
        Narrow the bracket based on function values at c and d.

        For MAXIMIZATION:
        - If f(c) > f(d): maximum in [a, d], narrow right
        - If f(d) >= f(c): maximum in [c, b], narrow left
        """
        if self.fc > self.fd:
            # Maximum in [a, d], narrow from right
            # Reuse c as new d
            self.b = self.d
            self.d = self.c
            self.fd = self.fc
            # Calculate new c
            self.c = self.a + (1 - GOLDEN_RATIO) * (self.b - self.a)
            self.fc = None
            self.phase = "eval_c"
        else:
            # Maximum in [c, b], narrow from left
            # Reuse d as new c
            self.a = self.c
            self.c = self.d
            self.fc = self.fd
            # Calculate new d
            self.d = self.a + GOLDEN_RATIO * (self.b - self.a)
            self.fd = None
            self.phase = "eval_d"

        self.current_step += 1

    def get_best_result(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return the position with the highest score."""
        if not self.evaluated_scores:
            return None, None

        best_idx = np.argmax(self.evaluated_scores)
        return self.evaluated_positions[best_idx], self.evaluated_scores[best_idx]

    def is_active(self) -> bool:
        """Check if golden section search should continue."""
        return self.active and self.current_step < self.max_iters
