# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from abc import ABC, abstractmethod

from gradient_free_optimizers._array_backend import mean


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
            Starting position in search space (integer indices).
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

        max_positions = self.optimizer.conv.max_positions

        for i, (d, o, max_pos) in enumerate(zip(direction, origin, max_positions)):
            if abs(d) < 1e-10:
                continue

            if d > 0:
                t_to_max = (max_pos - o) / d
                t_to_zero = -o / d
                max_t_positive = min(max_t_positive, t_to_max)
                max_t_negative = min(max_t_negative, -t_to_zero)
            else:
                t_to_zero = -o / d
                t_to_max = (max_pos - o) / d
                max_t_positive = min(max_t_positive, t_to_zero)
                max_t_negative = min(max_t_negative, -t_to_max)

        max_t = min(max_t_positive, max_t_negative)

        if max_t == float("inf"):
            # Direction is zero vector (shouldn't happen with normalized directions)
            # Fall back to average dimension size
            max_t = float(mean(max_positions))
        elif max_t < 0:
            # Numerical issue or at boundary corner
            max_t = 1.0

        # Ensure at least one grid step is possible (but respect actual bounds)
        # max_t can legitimately be small if we're near a boundary
        max_t = max(max_t, 1.0)

        return max_t

    def _snap_to_grid(self, position):
        """
        Snap a floating-point position to valid grid indices.

        Parameters
        ----------
        position : np.ndarray
            Position with potentially non-integer values.

        Returns
        -------
        np.ndarray
            Valid integer position within search space bounds.
        """
        return self.optimizer.conv2pos(position)

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
