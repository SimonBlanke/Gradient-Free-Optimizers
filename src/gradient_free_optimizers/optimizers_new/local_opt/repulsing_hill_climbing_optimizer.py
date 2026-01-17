# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Repulsing Hill Climbing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(inherits iteration methods from HillClimbingOptimizer)
"""

import numpy as np

from .hill_climbing_optimizer import HillClimbingOptimizer


class RepulsingHillClimbingOptimizer(HillClimbingOptimizer):
    """Hill Climbing with adaptive step size based on score improvement.

    When a worse solution is found, the step size is multiplied by the
    repulsion factor to escape the current region faster. When a better
    solution is found, the step size resets to normal.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    This "repulsion" mechanism helps escape local optima by taking
    progressively larger steps when stuck.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart to escape local optima.
    nth_process : int, optional
        Process index for parallel optimization.
    epsilon : float, default=0.03
        Base step size for generating neighbors (fraction of search space).
    distribution : str, default="normal"
        Distribution for step sizes: "normal", "laplace", or "logistic".
    n_neighbours : int, default=3
        Number of neighbors to evaluate before selecting the best.
    repulsion_factor : float, default=5
        Multiplier for step size when escaping worse regions.
    """

    name = "Repulsing Hill Climbing"
    _name_ = "repulsing_hill_climbing"
    __name__ = "RepulsingHillClimbingOptimizer"

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
        n_neighbours=3,
        repulsion_factor=5,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
        )
        self.repulsion_factor = repulsion_factor
        self.epsilon_mod = 1  # Multiplier for epsilon, increases when stuck

    # ═══════════════════════════════════════════════════════════════════════════
    # OVERRIDE BATCH METHODS: Use epsilon * epsilon_mod
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Generate new continuous values with adaptive step size.

        Uses epsilon * epsilon_mod to allow larger steps when stuck.
        """
        ranges = bounds[:, 1] - bounds[:, 0]
        effective_epsilon = self.epsilon * self.epsilon_mod
        sigmas = ranges * effective_epsilon

        noise_fn = self._DISTRIBUTIONS[self.distribution]
        noise = noise_fn(self._rng, sigmas, len(current))

        return current + noise

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Generate new categorical values with adaptive switch probability.

        Uses epsilon * epsilon_mod for higher switch probability when stuck.
        """
        n = len(current)
        effective_epsilon = self.epsilon * self.epsilon_mod

        # Higher epsilon_mod means more likely to switch categories
        switch_mask = self._rng.random(n) < effective_epsilon
        random_cats = np.floor(self._rng.random(n) * n_categories).astype(np.int64)

        return np.where(switch_mask, random_cats, current.astype(np.int64))

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Generate new discrete values with adaptive step size.

        Uses epsilon * epsilon_mod to allow larger steps when stuck.
        """
        max_positions = bounds[:, 1]
        effective_epsilon = self.epsilon * self.epsilon_mod
        sigmas = max_positions * effective_epsilon

        sigmas = np.maximum(sigmas, 1e-10)

        noise_fn = self._DISTRIBUTIONS[self.distribution]
        noise = noise_fn(self._rng, sigmas, len(current))

        return current + noise

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATE: Adjust epsilon_mod based on improvement
    # ═══════════════════════════════════════════════════════════════════════════

    def _evaluate(self, score_new):
        """Evaluate with adaptive step size adjustment.

        If the new score is worse than current, increase epsilon_mod
        to take larger steps and escape the current region.
        If the new score is better, reset epsilon_mod to normal.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # First, do the standard hill climbing evaluation
        super()._evaluate(score_new)

        # Then adjust epsilon_mod based on improvement
        if score_new <= self.score_current:
            # Worse score: increase step size to escape
            self.epsilon_mod = self.repulsion_factor
        else:
            # Better score: reset to normal step size
            self.epsilon_mod = 1
