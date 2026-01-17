# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
DIRECT (DIviding RECTangles) Algorithm.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
Uses Gower-like distance for mixed dimension types.
"""

import numpy as np

from ..smb_opt import SMBO


class DirectAlgorithm(SMBO):
    """DIRECT algorithm for global optimization.

    Dimension Support:
        - Continuous: YES (rectangle subdivision)
        - Categorical: YES (with Hamming distance)
        - Discrete: YES (normalized distance)

    Divides the search space into hyperrectangles and samples
    their centers, using a Lipschitz-based selection criterion.
    """

    name = "Direct Algorithm"
    _name_ = "direct_algorithm"
    __name__ = "DirectAlgorithm"

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
        """DIRECT uses rectangle centers, not iteration."""
        # TODO: Implement DIRECT-specific logic
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """DIRECT with categorical uses Hamming distance."""
        # TODO: Implement DIRECT-specific logic
        raise NotImplementedError("_iterate_categorical_batch() not yet implemented")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """DIRECT uses rectangle centers."""
        # TODO: Implement DIRECT-specific logic
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and update subspace Lipschitz bounds."""
        # TODO: Implement DIRECT evaluation
        raise NotImplementedError("evaluate() not yet implemented")
