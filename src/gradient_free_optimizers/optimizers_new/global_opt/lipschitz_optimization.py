# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Lipschitz Optimization.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

import numpy as np

from ..smb_opt import SMBO


class LipschitzOptimizer(SMBO):
    """Lipschitz-based global optimizer.

    Dimension Support:
        - Continuous: YES (Lipschitz bounds)
        - Categorical: YES (with appropriate distance metric)
        - Discrete: YES (Lipschitz bounds)

    Uses Lipschitz continuity assumptions to bound the objective
    function and guide the search.
    """

    name = "Lipschitz Optimizer"
    _name_ = "lipschitz_optimizer"
    __name__ = "LipschitzOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        max_sample_size=10000000,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            max_sample_size=max_sample_size,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Lipschitz-guided sampling for continuous dimensions."""
        # TODO: Implement Lipschitz-based iteration
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Lipschitz-guided sampling for categorical dimensions."""
        # TODO: Implement Lipschitz-based iteration
        raise NotImplementedError("_iterate_categorical_batch() not yet implemented")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Lipschitz-guided sampling for discrete dimensions."""
        # TODO: Implement Lipschitz-based iteration
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and update Lipschitz estimates."""
        # TODO: Implement Lipschitz evaluation
        raise NotImplementedError("evaluate() not yet implemented")
