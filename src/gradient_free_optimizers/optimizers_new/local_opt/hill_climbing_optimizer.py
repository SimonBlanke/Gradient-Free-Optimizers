# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Hill Climbing Optimizer with dimension-type-aware iteration.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class HillClimbingOptimizer(CoreOptimizer):
    """Hill Climbing optimizer using Gaussian noise for exploration.

    Dimension Support:
        - Continuous: YES (Gaussian noise scaled by range)
        - Categorical: YES (probabilistic category switching)
        - Discrete: YES (Gaussian noise, rounded to nearest index)

    The epsilon parameter controls the exploration intensity:
        - For continuous: sigma = range * epsilon
        - For categorical: switch_probability = epsilon
        - For discrete: sigma = max_index * epsilon
    """

    name = "Hill Climbing"
    _name_ = "hill_climbing"
    __name__ = "HillClimbingOptimizer"

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
        epsilon=0.03,
        distribution="normal",
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.epsilon = epsilon
        self.distribution = distribution

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS: Dimension-type-specific iteration
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Gaussian noise scaled by dimension range.

        Args:
            current: Current values, shape (n_continuous,)
            bounds: Min/max bounds, shape (n_continuous, 2)

        Returns
        -------
            New values with Gaussian noise added
        """
        # TODO: Implement vectorized Gaussian noise
        # ranges = bounds[:, 1] - bounds[:, 0]
        # sigmas = ranges * self.epsilon
        # noise = np.random.normal(0, sigmas)
        # return current + noise
        raise NotImplementedError("_iterate_continuous_batch() not yet implemented")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Probabilistic category switching.

        Args:
            current: Current category indices, shape (n_categorical,)
            n_categories: Number of categories per dimension, shape (n_categorical,)

        Returns
        -------
            New category indices (some may have switched)
        """
        # TODO: Implement vectorized category switching
        # n = len(current)
        # switch_mask = np.random.random(n) < self.epsilon
        # random_cats = np.floor(np.random.random(n) * n_categories).astype(int)
        # return np.where(switch_mask, random_cats, current)
        raise NotImplementedError("_iterate_categorical_batch() not yet implemented")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Gaussian noise for discrete dimensions.

        Args:
            current: Current positions, shape (n_discrete,)
            bounds: Min/max bounds, shape (n_discrete, 2)

        Returns
        -------
            New positions with Gaussian noise added (will be rounded later)
        """
        # TODO: Implement vectorized discrete iteration
        # max_positions = bounds[:, 1]
        # sigmas = max_positions * self.epsilon
        # noise = np.random.normal(0, sigmas)
        # return current + noise
        raise NotImplementedError("_iterate_discrete_batch() not yet implemented")

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATE
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate(self, score_new):
        """Evaluate and update if better (greedy hill climbing)."""
        # TODO: Implement greedy evaluation
        raise NotImplementedError("evaluate() not yet implemented")
