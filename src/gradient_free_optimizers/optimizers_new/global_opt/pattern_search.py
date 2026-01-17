# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Pattern Search (Hooke-Jeeves) Optimizer.

Supports: CONTINUOUS, DISCRETE_NUMERICAL
Note: Categorical dimensions require special handling.
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class PatternSearch(CoreOptimizer):
    """Pattern Search (Hooke-Jeeves) optimizer.

    Dimension Support:
        - Continuous: YES (coordinate-wise search)
        - Categorical: LIMITED (pattern moves don't apply naturally)
        - Discrete: YES (coordinate-wise search with integer steps)

    Uses exploratory moves along coordinate axes followed by
    pattern moves in successful directions.
    """

    name = "Pattern Search"
    _name_ = "pattern_search"
    __name__ = "PatternSearch"

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
        step_size=1.0,
        reduction_factor=0.5,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.step_size = step_size
        self.reduction_factor = reduction_factor

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Coordinate-wise exploratory moves for continuous dimensions."""
        # TODO: Implement pattern search moves
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Pattern search doesn't naturally apply to categorical dimensions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has limited support for categorical "
            f"dimensions. Pattern moves don't apply naturally to categories."
        )

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Coordinate-wise exploratory moves for discrete dimensions."""
        # TODO: Implement pattern search moves
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and update pattern."""
        # TODO: Implement pattern search evaluation
        raise NotImplementedError("evaluate() not yet implemented")
