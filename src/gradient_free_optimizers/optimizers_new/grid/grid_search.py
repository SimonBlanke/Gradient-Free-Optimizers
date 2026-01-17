# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Grid Search Optimizer.

Supports: DISCRETE_NUMERICAL, CATEGORICAL
Note: Continuous dimensions must be discretized for grid search.
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class GridSearchOptimizer(CoreOptimizer):
    """Exhaustive Grid Search optimizer.

    Dimension Support:
        - Continuous: LIMITED (must be discretized first)
        - Categorical: YES (enumerate all categories)
        - Discrete: YES (enumerate all values)

    Systematically evaluates all combinations of parameter values.
    """

    name = "Grid Search"
    _name_ = "grid_search"
    __name__ = "GridSearchOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.grid_index = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # GRID SEARCH ITERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def iterate(self):
        """Grid search iteration: move to next grid point."""
        # TODO: Implement grid enumeration
        raise NotImplementedError("iterate() not yet implemented")

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Grid search doesn't naturally apply to continuous dimensions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support continuous dimensions. "
            f"Use discretized values (np.linspace) instead of tuples."
        )

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Grid enumeration for categorical dimensions."""
        # TODO: Implement grid enumeration
        raise NotImplementedError("_iterate_categorical_batch() not yet implemented")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Grid enumeration for discrete dimensions."""
        # TODO: Implement grid enumeration
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and track best."""
        # TODO: Implement grid evaluation
        raise NotImplementedError("evaluate() not yet implemented")
