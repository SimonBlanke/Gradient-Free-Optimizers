# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class Direction:
    """
    Represents a search direction for Powell's method.

    This class handles movement along an arbitrary direction vector in the
    search space, supporting line searches for 1D optimization.
    """

    def __init__(self, direction_vector: np.ndarray):
        """
        Initialize a direction for line search.

        Parameters
        ----------
        direction_vector : np.ndarray
            The direction vector to search along. Will be normalized.
        """
        norm = np.linalg.norm(direction_vector)
        if norm < 1e-10:
            raise ValueError("Direction vector cannot be zero")
        self.direction = direction_vector / norm

    def get_position_at(self, origin: np.ndarray, t: float) -> np.ndarray:
        """
        Calculate position along the direction from an origin point.

        Parameters
        ----------
        origin : np.ndarray
            Starting position in search space
        t : float
            Step size along the direction (can be negative)

        Returns
        -------
        np.ndarray
            New position: origin + t * direction
        """
        return origin + t * self.direction

    @classmethod
    def from_two_points(cls, position_1: np.ndarray, position_2: np.ndarray):
        """
        Create a Direction from two points in the search space.

        Parameters
        ----------
        position_1 : np.ndarray
            Starting point
        position_2 : np.ndarray
            End point

        Returns
        -------
        Direction
            Direction pointing from position_1 to position_2
        """
        direction_vector = position_2 - position_1
        return cls(direction_vector)

    @classmethod
    def coordinate_axis(cls, dimension: int, n_dimensions: int):
        """
        Create a Direction along a coordinate axis.

        Parameters
        ----------
        dimension : int
            Which axis (0-indexed)
        n_dimensions : int
            Total number of dimensions

        Returns
        -------
        Direction
            Unit vector along the specified axis
        """
        direction_vector = np.zeros(n_dimensions)
        direction_vector[dimension] = 1.0
        return cls(direction_vector)
