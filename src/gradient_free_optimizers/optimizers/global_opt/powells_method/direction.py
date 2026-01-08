# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._array_backend import array, linalg, zeros


class Direction:
    """
    Represents a search direction for Powell's method.

    This class handles movement along an arbitrary direction vector in the
    search space, supporting line searches for 1D optimization.
    """

    def __init__(self, direction_vector):
        """
        Initialize a direction for line search.

        Parameters
        ----------
        direction_vector : array-like
            The direction vector to search along. Will be normalized.
        """
        direction_vector = array(direction_vector)
        norm = linalg.norm(direction_vector)
        if norm < 1e-10:
            raise ValueError("Direction vector cannot be zero")
        self.direction = direction_vector / norm

    def get_position_at(self, origin, t: float):
        """
        Calculate position along the direction from an origin point.

        Parameters
        ----------
        origin : array-like
            Starting position in search space
        t : float
            Step size along the direction (can be negative)

        Returns
        -------
        array
            New position: origin + t * direction
        """
        return array(origin) + t * self.direction

    @classmethod
    def from_two_points(cls, position_1, position_2):
        """
        Create a Direction from two points in the search space.

        Parameters
        ----------
        position_1 : array-like
            Starting point
        position_2 : array-like
            End point

        Returns
        -------
        Direction
            Direction pointing from position_1 to position_2
        """
        direction_vector = array(position_2) - array(position_1)
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
        direction_vector = zeros(n_dimensions)
        direction_vector[dimension] = 1.0
        return cls(direction_vector)
