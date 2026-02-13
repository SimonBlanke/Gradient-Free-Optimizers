# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Random Restart Hill Climbing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(inherits iteration methods from HillClimbingOptimizer)
"""

import numpy as np

from ..local_opt import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    """Hill Climbing with periodic random restarts.

    Periodically restarts from a random position to escape local optima
    and explore different regions of the search space.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    Every n_iter_restart iterations, the optimizer jumps to a completely
    random position instead of climbing from the current position.

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
    n_iter_restart : int, default=10
        Number of iterations between random restarts.
    """

    name = "Random Restart Hill Climbing"
    _name_ = "random_restart_hill_climbing"
    __name__ = "RandomRestartHillClimbingOptimizer"

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
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        n_iter_restart=10,
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
        self.n_iter_restart = n_iter_restart

    def _generate_position(self):
        """Generate a position - either climbing or random restart.

        Every n_iter_restart iterations, generate a completely random
        position instead of climbing from current.

        Returns
        -------
            Clipped position as numpy array
        """
        # Check if this is a restart iteration
        not_zero = self.nth_trial != 0
        mod_zero = self.nth_trial % self.n_iter_restart == 0

        if not_zero and mod_zero:
            # Random restart - generate completely random position
            return self._generate_random_position()
        else:
            # Normal hill climbing
            return super()._generate_position()

    def _generate_random_position(self):
        """Generate a completely random position (like RandomSearch).

        Returns
        -------
            Clipped random position as numpy array
        """
        n_dims = len(self.search_space)
        new_pos = np.empty(n_dims)

        # Random continuous values
        if self._continuous_mask is not None and self._continuous_mask.any():
            mins = self._continuous_bounds[:, 0]
            maxs = self._continuous_bounds[:, 1]
            new_pos[self._continuous_mask] = self._rng.uniform(mins, maxs)

        # Random categorical indices
        if self._categorical_mask is not None and self._categorical_mask.any():
            n_cats = self._categorical_sizes
            new_pos[self._categorical_mask] = np.floor(
                self._rng.random(len(n_cats)) * n_cats
            ).astype(np.int64)

        # Random discrete indices
        if self._discrete_mask is not None and self._discrete_mask.any():
            mins = self._discrete_bounds[:, 0].astype(np.int64)
            maxs = self._discrete_bounds[:, 1].astype(np.int64)
            new_pos[self._discrete_mask] = self._rng.integers(mins, maxs + 1)

        return self._clip_position(new_pos)
