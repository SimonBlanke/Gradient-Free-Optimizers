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
        Step size for generating neighbors (fraction of search space).
    distribution : str, default="normal"
        Distribution for step sizes: "normal", "laplace", or "logistic".
    n_neighbours : int, default=3
        Number of neighbors to evaluate before selecting the best.
    """

    name = "Hill Climbing"
    _name_ = "hill_climbing"
    __name__ = "HillClimbingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    # Distribution functions for noise generation
    _DISTRIBUTIONS = {
        "normal": lambda rng, scale, size: rng.normal(0, scale, size),
        "laplace": lambda rng, scale, size: rng.laplace(0, scale, size),
        "logistic": lambda rng, scale, size: rng.logistic(0, scale, size),
        "gumbel": lambda rng, scale, size: rng.gumbel(0, scale, size),
        "uniform": lambda rng, scale, size: rng.uniform(-scale, scale, size),
    }

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
        self.n_neighbours = n_neighbours

        # Initialize RNG for reproducibility using the actual seed
        # (self.random_seed is set by CoreOptimizer and accounts for nth_process)
        self._rng = np.random.default_rng(self.random_seed)

        # Validate distribution parameter
        if distribution not in self._DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                f"Choose from: {list(self._DISTRIBUTIONS.keys())}"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS: Dimension-type-specific iteration
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Generate new continuous values using Gaussian noise scaled by range.

        The noise magnitude is proportional to the dimension's range,
        ensuring consistent exploration behavior regardless of scale.

        Args:
            current: Current values, shape (n_continuous,)
            bounds: Min/max bounds, shape (n_continuous, 2)
                    bounds[:, 0] = min values
                    bounds[:, 1] = max values

        Returns
        -------
            New values with noise added (not yet clipped to bounds)
        """
        # Calculate range for each dimension
        ranges = bounds[:, 1] - bounds[:, 0]

        # Scale sigma by range and epsilon
        sigmas = ranges * self.epsilon

        # Generate noise using the configured distribution
        noise_fn = self._DISTRIBUTIONS[self.distribution]
        noise = noise_fn(self._rng, sigmas, len(current))

        return current + noise

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Generate new categorical values using probabilistic switching.

        With probability epsilon, switch to a random category.
        Otherwise, keep the current category.

        Args:
            current: Current category indices, shape (n_categorical,)
            n_categories: Number of categories per dimension, shape (n_categorical,)

        Returns
        -------
            New category indices (integers)
        """
        n = len(current)

        # Determine which dimensions will switch (Bernoulli trial)
        switch_mask = self._rng.random(n) < self.epsilon

        # Generate random categories for switching dimensions
        # Use uniform distribution over [0, n_categories)
        random_cats = np.floor(self._rng.random(n) * n_categories).astype(np.int64)

        # Apply switch: use random if switching, otherwise keep current
        return np.where(switch_mask, random_cats, current.astype(np.int64))

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Generate new discrete values using Gaussian noise.

        Similar to continuous, but operates on discrete indices.
        The result will be rounded to integers by _clip_position.

        Args:
            current: Current positions (indices), shape (n_discrete,)
            bounds: Min/max bounds, shape (n_discrete, 2)
                    bounds[:, 0] = min index (typically 0)
                    bounds[:, 1] = max index

        Returns
        -------
            New positions with noise added (float, will be rounded)
        """
        # Use max position to scale sigma (similar to continuous)
        max_positions = bounds[:, 1]
        sigmas = max_positions * self.epsilon

        # Prevent zero sigma for single-value dimensions
        sigmas = np.maximum(sigmas, 1e-10)

        # Generate noise using the configured distribution
        noise_fn = self._DISTRIBUTIONS[self.distribution]
        noise = noise_fn(self._rng, sigmas, len(current))

        return current + noise

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATE: Greedy Hill Climbing with n_neighbours
    # ═══════════════════════════════════════════════════════════════════════════

    def _evaluate(self, score_new):
        """Greedy selection after n_neighbours trials.

        Hill climbing evaluates n_neighbours positions, then moves to the
        best one among them. This multi-sample approach reduces the
        probability of missing good directions in noisy landscapes.

        Note: score tracking is already done by CoreOptimizer.evaluate()
        before this method is called.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Every n_neighbours trials, select the best among recent samples
        if self.nth_trial % self.n_neighbours == 0:
            # Get the last n_neighbours scores and positions
            recent_scores = self.scores_valid[-self.n_neighbours :]
            recent_positions = self.positions_valid[-self.n_neighbours :]

            # Find the best among recent samples
            best_idx = np.argmax(recent_scores)
            best_score = recent_scores[best_idx]
            best_pos = recent_positions[best_idx]

            # Update current position to best found
            self._update_current(best_pos, best_score)

            # Update global best if this is better
            self._update_best(best_pos, best_score)
