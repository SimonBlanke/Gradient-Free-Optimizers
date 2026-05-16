# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Spiral class for Spiral Optimization Algorithm.

A Spiral particle moves in a spiral trajectory toward the center position,
combining rotation and contraction to balance exploration and exploitation.
"""

from __future__ import annotations

import math

from gradient_free_optimizers._array_backend import array, zeros
from gradient_free_optimizers._dimension_types import DimensionType

from ..local_opt import HillClimbingOptimizer


def rotation(n_dim, vector, rotation_degrees=90.0):
    """Rotate a vector by a configurable angle.

    In two dimensions this is the usual planar rotation. In higher dimensions,
    the vector is rotated in a deterministic plane spanned by itself and a
    perpendicular direction derived from the least-aligned coordinate axis.

    A one-dimensional vector has no rotation plane, so it keeps the historical
    sign-flip behavior.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    vector : np.ndarray
        Vector to rotate.
    rotation_degrees : float, default=90.0
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated vector.
    """
    vector = array(vector)

    if n_dim == 1:
        return -vector

    angle = math.radians(rotation_degrees)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    if n_dim == 2:
        x, y = float(vector[0]), float(vector[1])
        return array(
            [
                cos_angle * x - sin_angle * y,
                sin_angle * x + cos_angle * y,
            ]
        )

    norm_sq = sum(float(vector[i]) ** 2 for i in range(n_dim))
    if norm_sq == 0:
        return zeros(n_dim)

    norm = math.sqrt(norm_sq)
    unit = [float(vector[i]) / norm for i in range(n_dim)]
    anchor_idx = min(range(n_dim), key=lambda idx: abs(unit[idx]))

    projection = unit[anchor_idx]
    perpendicular = [
        (1.0 if i == anchor_idx else 0.0) - projection * unit[i] for i in range(n_dim)
    ]
    perpendicular_norm = math.sqrt(sum(value**2 for value in perpendicular))
    if perpendicular_norm == 0:
        return array([cos_angle * float(vector[i]) for i in range(n_dim)])

    perpendicular_unit = [value / perpendicular_norm for value in perpendicular]
    return array(
        [
            norm * (cos_angle * unit[i] + sin_angle * perpendicular_unit[i])
            for i in range(n_dim)
        ]
    )


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
    rotation_degrees : float, default=90.0
        Rotation angle applied to the normalized offset at each step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.decay_rate = None
        self.spiral_radius = 1.0
        self.rotation_degrees = 90.0
        self.decay_factor = self.spiral_radius

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
            pos_new = (array(pos) + array(velo)).astype(int)
            return self._clip_position(pos_new)

        # Type-aware movement for mixed dimension types
        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            new_val = pos[idx] + velo[idx]
            pos_new.append(new_val)

        return self._conv2pos_typed(array(pos_new))

    def move_spiral(self, center_pos):
        """Move particle in spiral pattern toward center.

        The spiral equation is:
            new_pos = denormalize(
                center_norm + radius * decay * rotation(current_norm - center_norm)
            )

        The decay factor contracts the spiral over time, while
        the configured rotation angle creates the spiral trajectory.

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

        center = array(center_pos)
        current = array(self._pos_current)
        center_norm = self._normalize_position(center)
        current_norm = self._normalize_position(current)

        rot = rotation(
            len(center_norm),
            current_norm - center_norm,
            rotation_degrees=self.rotation_degrees,
        )
        new_norm = center_norm + self.decay_factor * rot
        new_pos = self._denormalize_position(new_norm)

        # Convert to valid position
        pos_new = self._conv2pos_typed(new_pos)

        # Track position (property setter auto-appends)
        self._pos_new = pos_new

        return pos_new

    def _compute_dimension_bounds(self):
        lower = zeros(len(self.search_space))
        upper = zeros(len(self.search_space))

        for i, info in enumerate(self.conv.dim_infos):
            if info.dim_type.is_continuous_like:
                lower[i] = info.bounds[0]
                upper[i] = info.bounds[1]
            elif info.dim_type == DimensionType.CATEGORICAL:
                lower[i] = 0
                upper[i] = info.size - 1
            else:
                lower[i] = 0
                upper[i] = info.size - 1

        return lower, upper

    def _normalize_position(self, position):
        lower, upper = self._compute_dimension_bounds()
        normalized = []

        for value, lo, hi in zip(position, lower, upper):
            span = hi - lo
            if span > 0:
                normalized.append((value - lo) / span)
            else:
                normalized.append(0.0)

        return array(normalized)

    def _denormalize_position(self, normalized_position):
        lower, upper = self._compute_dimension_bounds()
        position = []

        for value, lo, hi in zip(normalized_position, lower, upper):
            span = hi - lo
            if span > 0:
                position.append(lo + value * span)
            else:
                position.append(lo)

        return array(position)

    def _conv2pos_typed(self, pos):
        """Convert position array to valid position with proper types.

        Clips values to bounds and ensures correct data types for each dimension.
        """
        return self._clip_position(pos)

    def _on_evaluate(self, score_new):
        """Evaluate and track scores."""
        # Update current position tracking
        self._pos_current = self._pos_new.copy() if self._pos_new is not None else None
        self._score_current = score_new

        # Update best if improved
        if self._pos_best is None or score_new > self._score_best:
            self._pos_best = self._pos_new.copy() if self._pos_new is not None else None
            self._score_best = score_new
