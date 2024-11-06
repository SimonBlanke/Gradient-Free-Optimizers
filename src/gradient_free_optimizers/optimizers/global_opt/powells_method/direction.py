# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class Direction:
    def __init__(self, position_1, position_2) -> None:
        """
        Parameters:
        - position_1: dict, starting point in the search-space
        - position_2: dict, end point in the search-space
        """

        self.position_1 = position_1
        self.position_2 = position_2

    def get_new_position(self, t):
        """
        Calculate a position on the line (vector) between two positions using parameter t.

        Parameters:
        - t: float, parameter indicating the position along the line (0 <= t <= 1 for within the line segment)

        Returns:
        - dict representing the new position in the search-space coordinate system
        """
        new_position = {}
        for dim in self.position_1:
            # Calculate the position along the line in each dimension
            new_position[dim] = self.position_1[dim] + t * (
                self.position_2[dim] - self.position_1[dim]
            )
        return new_position
