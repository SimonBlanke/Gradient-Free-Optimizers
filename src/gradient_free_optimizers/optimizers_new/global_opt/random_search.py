# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Random Search Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class RandomSearchOptimizer(CoreOptimizer):
    """Random Search optimizer - samples randomly from the search space.

    Dimension Support:
        - Continuous: YES (uniform random in range)
        - Categorical: YES (uniform random category)
        - Discrete: YES (uniform random index)

    Random search provides a baseline and can be surprisingly effective
    for low-dimensional or highly multimodal problems.
    """

    name = "Random Search"
    _name_ = "random_search"
    __name__ = "RandomSearchOptimizer"

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

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Uniform random sampling in continuous ranges."""
        # TODO: Implement
        # return np.random.uniform(bounds[:, 0], bounds[:, 1])
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Uniform random category selection."""
        # TODO: Implement
        # return np.floor(np.random.random(len(current)) * n_categories).astype(int)
        raise NotImplementedError("_iterate_categorical_batch() not yet implemented")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Uniform random index selection."""
        # TODO: Implement
        # return np.random.randint(bounds[:, 0], bounds[:, 1] + 1)
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate and update best if improved."""
        # TODO: Implement
        raise NotImplementedError("evaluate() not yet implemented")
