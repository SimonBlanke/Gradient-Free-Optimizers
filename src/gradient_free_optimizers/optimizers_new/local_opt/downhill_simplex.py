# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Downhill Simplex (Nelder-Mead) Optimizer.

Supports: CONTINUOUS, DISCRETE_NUMERICAL
Note: Categorical dimensions require special handling in simplex methods.
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class DownhillSimplexOptimizer(CoreOptimizer):
    """Downhill Simplex (Nelder-Mead) optimizer.

    Dimension Support:
        - Continuous: YES (native simplex operations)
        - Categorical: LIMITED (simplex operations don't naturally apply)
        - Discrete: YES (with rounding)

    The Nelder-Mead simplex method uses geometric operations
    (reflection, expansion, contraction) on a simplex of n+1 points.
    """

    name = "Downhill Simplex"
    _name_ = "downhill_simplex"
    __name__ = "DownhillSimplexOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        alpha=1.0,
        gamma=2.0,
        beta=0.5,
        sigma=0.5,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.alpha = alpha  # Reflection coefficient
        self.gamma = gamma  # Expansion coefficient
        self.beta = beta  # Contraction coefficient
        self.sigma = sigma  # Shrink coefficient

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Simplex operations for continuous dimensions."""
        # TODO: Implement simplex reflection/expansion/contraction
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Categorical dimensions in simplex are challenging.

        Simplex methods rely on continuous geometric operations which
        don't naturally apply to categorical dimensions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has limited support for categorical "
            f"dimensions. Simplex operations don't naturally apply to categories."
        )

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Simplex operations for discrete dimensions (with rounding)."""
        # TODO: Implement simplex operations with rounding
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and update simplex."""
        # TODO: Implement simplex update logic
        raise NotImplementedError("evaluate() not yet implemented")
