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

    Random search is the simplest optimization strategy that samples positions
    uniformly at random from the search space. Despite its simplicity, it can
    be surprisingly effective for many problems, especially in high dimensions.

    Dimension Support:
        - Continuous: YES (uniform random in range)
        - Categorical: YES (uniform random category)
        - Discrete: YES (uniform random index)

    The algorithm has no memory of previous evaluations and makes no assumptions
    about the objective function. Each iteration independently samples a random
    point, making it trivially parallelizable.

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
        Probability of random restart (irrelevant for random search).
    nth_process : int, optional
        Process index for parallel optimization.
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

        # Initialize RNG for reproducibility using the actual seed
        # (self.random_seed is set by CoreOptimizer and accounts for nth_process)
        self._rng = np.random.default_rng(self.random_seed)

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Uniform random sampling in continuous ranges.

        Accesses via: self._continuous_bounds

        Random search ignores current position - each sample is independent.

        Returns
        -------
        np.ndarray
            Random values uniformly distributed in [min, max]
        """
        bounds = self._continuous_bounds
        mins = bounds[:, 0]
        maxs = bounds[:, 1]
        return self._rng.uniform(mins, maxs)

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Uniform random category selection.

        Accesses via: self._categorical_sizes

        Random search ignores current position - each sample is independent.

        Returns
        -------
        np.ndarray
            Random category indices
        """
        n_categories = self._categorical_sizes
        n = len(n_categories)
        return np.floor(self._rng.random(n) * n_categories).astype(np.int64)

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Uniform random index selection.

        Accesses via: self._discrete_bounds

        Random search ignores current position - each sample is independent.

        Returns
        -------
        np.ndarray
            Random indices within bounds
        """
        bounds = self._discrete_bounds
        mins = bounds[:, 0].astype(np.int64)
        maxs = bounds[:, 1].astype(np.int64)
        # randint is exclusive on high, so add 1
        return self._rng.integers(mins, maxs + 1)

    def _evaluate(self, score_new):
        """Update best position if this score is better.

        Random search has no concept of "current" position since each
        iteration is independent. We only track the global best.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Simply update best if this is better
        self._update_best(self.pos_new, score_new)
