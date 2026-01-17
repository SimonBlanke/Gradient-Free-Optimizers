# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Orthogonal Grid Search (Latin Hypercube-like).

Supports: DISCRETE_NUMERICAL, CATEGORICAL
"""

from .grid_search import GridSearchOptimizer


class OrthogonalGridSearch(GridSearchOptimizer):
    """Orthogonal Grid Search - Latin Hypercube-like sampling.

    Dimension Support:
        - Continuous: LIMITED (must be discretized)
        - Categorical: YES
        - Discrete: YES

    Uses orthogonal array sampling to achieve better coverage
    with fewer evaluations than full grid search.
    """

    name = "Orthogonal Grid Search"
    _name_ = "orthogonal_grid_search"
    __name__ = "OrthogonalGridSearch"

    def iterate(self):
        """Orthogonal grid search iteration."""
        # TODO: Implement orthogonal sampling
        raise NotImplementedError("iterate() not yet implemented")
