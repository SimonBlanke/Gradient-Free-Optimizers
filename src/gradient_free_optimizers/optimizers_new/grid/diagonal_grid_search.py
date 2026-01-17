# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Diagonal Grid Search.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
"""

from .grid_search import GridSearchOptimizer


class DiagonalGridSearch(GridSearchOptimizer):
    """Diagonal Grid Search - searches along diagonals.

    Dimension Support:
        - Continuous: LIMITED (must be discretized)
        - Categorical: YES
        - Discrete: YES

    Searches the grid along diagonal paths rather than
    row-by-row enumeration.
    """

    name = "Diagonal Grid Search"
    _name_ = "diagonal_grid_search"
    __name__ = "DiagonalGridSearch"

    def iterate(self):
        """Diagonal grid search iteration."""
        # TODO: Implement diagonal traversal
        raise NotImplementedError("iterate() not yet implemented")
