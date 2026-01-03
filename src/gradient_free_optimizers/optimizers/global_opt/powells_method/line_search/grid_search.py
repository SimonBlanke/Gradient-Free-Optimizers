# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Optional, Tuple, List

import numpy as np

from .base import LineSearch


class GridLineSearch(LineSearch):
    """
    Grid-based line search strategy.

    Evaluates positions at evenly spaced intervals along the search direction.
    Simple and thorough, but may require more evaluations than adaptive methods.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.grid_positions: List[np.ndarray] = []
        self.evaluated_positions: List[np.ndarray] = []
        self.evaluated_scores: List[float] = []
        self.current_step: int = 0
        self.active: bool = False

    def start(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_iters: int,
    ) -> None:
        """Generate grid positions along the direction."""
        self.current_step = 0
        self.evaluated_positions = []
        self.evaluated_scores = []
        self.active = True

        self.grid_positions = self._generate_grid_positions(
            origin, direction, max_iters
        )

    def _generate_grid_positions(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        n_steps: int,
    ) -> List[np.ndarray]:
        """
        Generate evenly spaced positions along the direction.

        Parameters
        ----------
        origin : np.ndarray
            Starting position.
        direction : np.ndarray
            Normalized direction vector.
        n_steps : int
            Number of positions to generate.

        Returns
        -------
        List[np.ndarray]
            List of valid positions along the line.
        """
        max_t = self._compute_max_step(origin, direction)

        positions = []
        t_values = np.linspace(-max_t, max_t, n_steps)

        for t in t_values:
            pos_float = origin + t * direction
            pos_valid = self._snap_to_grid(pos_float)

            # Avoid duplicates
            is_duplicate = any(np.array_equal(pos_valid, p) for p in positions)
            if not is_duplicate and self._is_valid(pos_valid):
                positions.append(pos_valid)

        return positions

    def get_next_position(self) -> Optional[np.ndarray]:
        """Return the next grid position to evaluate."""
        if self.current_step < len(self.grid_positions):
            pos = self.grid_positions[self.current_step]
            self.current_step += 1
            return pos
        else:
            self.active = False
            return None

    def update(self, position: np.ndarray, score: float) -> None:
        """Record the evaluation result."""
        self.evaluated_positions.append(position.copy())
        self.evaluated_scores.append(score)

    def get_best_result(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return the position with the highest score."""
        if not self.evaluated_scores:
            return None, None

        best_idx = np.argmax(self.evaluated_scores)
        return self.evaluated_positions[best_idx], self.evaluated_scores[best_idx]

    def is_active(self) -> bool:
        """Check if more grid positions need evaluation."""
        return self.active and self.current_step < len(self.grid_positions)
