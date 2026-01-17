# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Sequential Model-Based Optimization (SMBO) base class.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class SMBO(CoreOptimizer):
    """Base class for Sequential Model-Based Optimization.

    Dimension Support:
        - Continuous: YES (surrogate model based)
        - Categorical: YES (with appropriate encoding)
        - Discrete: YES (surrogate model based)

    SMBO algorithms build a surrogate model of the objective function
    and use an acquisition function to select the next point to evaluate.
    """

    name = "SMBO"
    _name_ = "smbo"
    __name__ = "SMBO"

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
        )
        self.max_sample_size = max_sample_size

        # Surrogate model state
        self.surrogate_model = None
        self.X_sample = None
        self.y_sample = None

    # ═══════════════════════════════════════════════════════════════════════════
    # SMBO-SPECIFIC ITERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def iterate(self):
        """SMBO iteration: fit surrogate, optimize acquisition."""
        # TODO: Implement surrogate model fitting and acquisition optimization
        raise NotImplementedError("iterate() not yet implemented")

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Acquisition-based sampling for continuous dimensions."""
        # TODO: Implement acquisition optimization
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Acquisition-based sampling for categorical dimensions."""
        # TODO: Implement acquisition optimization with encoding
        raise NotImplementedError("_iterate_categorical_batch() not yet implemented")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Acquisition-based sampling for discrete dimensions."""
        # TODO: Implement acquisition optimization
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and add to training data."""
        # TODO: Implement SMBO evaluation
        raise NotImplementedError("evaluate() not yet implemented")
