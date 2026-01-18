# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Abstract base class for line search strategies in Powell's method.
"""

from abc import ABC, abstractmethod

import numpy as np


class LineSearch(ABC):
    """
    Abstract base class for line search strategies in Powell's method.

    A line search finds the optimum along a given direction from a starting point.
    Different strategies (grid, golden section, hill climbing) implement this
    interface to provide various trade-offs between exploration and efficiency.
    """

    def __init__(self, optimizer):
        """
        Initialize the line search strategy.

        Parameters
        ----------
        optimizer : PowellsMethod
            Reference to the parent optimizer for accessing converter and utilities.
        """
        self.optimizer = optimizer

    @abstractmethod
    def start(
        self,
        origin,
        direction,
        max_iters: int,
    ) -> None:
        """
        Initialize a new line search along a direction.

        Parameters
        ----------
        origin : np.ndarray
            Starting position in search space.
        direction : np.ndarray
            Normalized direction vector to search along.
        max_iters : int
            Maximum number of evaluations for this line search.
        """
        pass

    @abstractmethod
    def get_next_position(self):
        """
        Get the next position to evaluate.

        Returns
        -------
        Optional[np.ndarray]
            Next position to evaluate, or None if line search is complete.
        """
        pass

    @abstractmethod
    def update(self, position, score: float) -> None:
        """
        Update the line search state after an evaluation.

        Parameters
        ----------
        position : np.ndarray
            The position that was evaluated.
        score : float
            The score obtained at that position.
        """
        pass

    @abstractmethod
    def get_best_result(self) -> tuple:
        """
        Get the best result found during this line search.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[float]]
            (best_position, best_score) or (None, None) if no valid results.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """
        Check if the line search is still in progress.

        Returns
        -------
        bool
            True if more evaluations are needed, False if complete.
        """
        pass

    def _compute_max_step(self, origin, direction) -> float:
        """
        Compute the maximum step size along a direction that stays within bounds.

        Parameters
        ----------
        origin : np.ndarray
            Starting position.
        direction : np.ndarray
            Normalized direction vector.

        Returns
        -------
        float
            Maximum step size (for symmetric bounds [-max_t, max_t]).
        """
        max_t_positive = float("inf")
        max_t_negative = float("inf")

        # Get dimension bounds
        dim_names = list(self.optimizer.search_space.keys())
        for i, name in enumerate(dim_names):
            d = direction[i]
            o = origin[i]

            if abs(d) < 1e-10:
                continue

            dim_def = self.optimizer.search_space[name]
            if isinstance(dim_def, tuple):
                # Continuous
                min_val, max_val = dim_def
            elif isinstance(dim_def, (list, np.ndarray)):
                # Categorical or discrete: index bounds
                min_val, max_val = 0, len(dim_def) - 1
            else:
                continue

            if d > 0:
                t_to_max = (max_val - o) / d
                t_to_min = (min_val - o) / d
                max_t_positive = min(max_t_positive, t_to_max)
                max_t_negative = min(max_t_negative, -t_to_min)
            else:
                t_to_min = (min_val - o) / d
                t_to_max = (max_val - o) / d
                max_t_positive = min(max_t_positive, t_to_min)
                max_t_negative = min(max_t_negative, -t_to_max)

        max_t = min(max_t_positive, max_t_negative)

        if max_t == float("inf"):
            # Fallback
            max_t = 10.0
        elif max_t < 0:
            max_t = 1.0

        max_t = max(max_t, 1.0)

        return max_t

    def _snap_to_grid(self, position):
        """
        Snap a floating-point position to valid grid/bounds.

        Parameters
        ----------
        position : np.ndarray
            Position with potentially non-integer values.

        Returns
        -------
        np.ndarray
            Valid position within search space bounds.
        """
        return self.optimizer.conv2pos_typed(position)

    def _is_valid(self, position) -> bool:
        """
        Check if a position satisfies constraints.

        Parameters
        ----------
        position : np.ndarray
            Position to check.

        Returns
        -------
        bool
            True if position is valid (not in constraint).
        """
        return self.optimizer.conv.not_in_constraint(position)
