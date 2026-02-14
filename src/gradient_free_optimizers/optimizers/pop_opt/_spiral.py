# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Spiral class for Spiral Optimization Algorithm.

A Spiral particle moves in a spiral trajectory toward the center position,
combining rotation and contraction to balance exploration and exploitation.
"""

import numpy as np

from ..local_opt import HillClimbingOptimizer


def rotation(n_dim, vector):
    """Build rotation matrix and apply to vector.

    Creates a rotation matrix R of shape (n_dim, n_dim) where:
    - Identity shifted down by one row (bottom-left block)
    - -1 in top-right corner

    This creates a cyclic permutation with sign flip, causing
    spiral-like behavior when applied repeatedly.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    vector : np.ndarray
        Vector to rotate.

    Returns
    -------
    np.ndarray
        Rotated vector.
    """
    if n_dim == 1:
        return np.array([-1.0])  # Return as array for consistency

    # Build rotation matrix
    # R has shape (n_dim, n_dim)
    # It's the identity matrix (n_dim-1 x n_dim-1) padded with zeros:
    # - 1 row of zeros on top
    # - 1 column of zeros on right
    # - then R[0, n_dim-1] = -1
    R = np.zeros((n_dim, n_dim))
    for i in range(n_dim - 1):
        R[i + 1, i] = 1.0
    R[0, n_dim - 1] = -1.0

    return np.matmul(R, np.array(vector))


class Spiral(HillClimbingOptimizer):
    """Individual spiral particle in Spiral Optimization.

    Each spiral particle maintains:
    - Position (inherited from HillClimbingOptimizer)
    - Decay factor (controls spiral contraction)
    - Reference to global best position

    The spiral movement combines:
    - Rotation around the center (global best)
    - Contraction toward the center

    Parameters
    ----------
    search_space : dict
        Search space definition.
    decay_rate : float, default=0.99
        Rate at which spiral radius contracts per iteration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.decay_rate = None
        self.decay_factor = 3  # Initial scaling factor

    def _move_part(self, pos, velo):
        """Apply velocity to position.

        Parameters
        ----------
        pos : np.ndarray
            Current position.
        velo : np.ndarray
            Velocity/displacement to apply.

        Returns
        -------
        np.ndarray
            New position after applying velocity.
        """
        if self.conv.is_legacy_mode:
            pos_new = (np.array(pos) + np.array(velo)).astype(int)
            n_zeros = [0] * len(self.conv.max_positions)
            return np.clip(pos_new, n_zeros, self.conv.max_positions)

        # Type-aware movement for mixed dimension types
        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            new_val = pos[idx] + velo[idx]
            pos_new.append(new_val)

        return self._conv2pos_typed(np.array(pos_new))

    def move_spiral(self, center_pos):
        """Move particle in spiral pattern toward center.

        The spiral equation is:
            new_pos = center + decay * rotation(current - center)

        The decay factor contracts the spiral over time, while
        the rotation creates the spiral trajectory.

        Parameters
        ----------
        center_pos : np.ndarray
            Center position (typically global best).

        Returns
        -------
        np.ndarray
            New position after spiral movement.
        """
        # Guard against None positions during early iterations
        if center_pos is None or self._pos_current is None:
            # Fall back to random move during early iterations
            pos_new = self.init.move_random_typed()
            self._pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Update decay factor
        self.decay_factor *= self.decay_rate

        # Compute step rate based on search space size
        if self.conv.is_legacy_mode:
            step_rate = self.decay_factor * np.array(self.conv.max_positions) / 1000
        else:
            # For typed dimensions, use bounds to compute scale
            scales = []
            from gradient_free_optimizers._dimension_types import DimensionType

            for idx, dim_type in enumerate(self.conv.dim_types):
                bounds = self.conv.dim_infos[idx].bounds
                if dim_type == DimensionType.CONTINUOUS:
                    scales.append(bounds[1] - bounds[0])
                else:
                    scales.append(bounds[1])
            step_rate = self.decay_factor * np.array(scales) / 1000

        # Compute rotated offset from center
        A = np.array(center_pos)
        rot = rotation(len(center_pos), np.array(self._pos_current) - A)

        # Combine rotation with decay
        B = step_rate * rot
        new_pos = A + B

        # Convert to valid position
        if self.conv.is_legacy_mode:
            n_zeros = [0] * len(self.conv.max_positions)
            pos_new = np.clip(new_pos, n_zeros, self.conv.max_positions).astype(int)
        else:
            pos_new = self._conv2pos_typed(new_pos)

        # Track position (property setter auto-appends)
        self._pos_new = pos_new

        return pos_new

    def _conv2pos_typed(self, pos):
        """Convert position array to valid position with proper types.

        Clips values to bounds and ensures correct data types for each dimension.
        """
        if self.conv.is_legacy_mode:
            n_zeros = [0] * len(self.conv.max_positions)
            return np.clip(pos, n_zeros, self.conv.max_positions).astype(int)

        from gradient_free_optimizers._dimension_types import DimensionType

        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            bounds = self.conv.dim_infos[idx].bounds
            val = pos[idx]

            if dim_type == DimensionType.CONTINUOUS:
                # Clip to bounds, keep as float
                pos_new.append(np.clip(val, bounds[0], bounds[1]))
            else:
                # Discrete or categorical: clip and convert to int
                pos_new.append(int(np.clip(round(val), bounds[0], bounds[1])))

        return np.array(pos_new)

    def _on_evaluate(self, score_new):
        """Evaluate and track scores."""
        # Update current position tracking
        self._pos_current = self._pos_new.copy() if self._pos_new is not None else None
        self._score_current = score_new

        # Update best if improved
        if self._pos_best is None or score_new > self._score_best:
            self._pos_best = self._pos_new.copy() if self._pos_new is not None else None
            self._score_best = score_new
