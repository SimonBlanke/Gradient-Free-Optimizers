# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Random Annealing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(inherits iteration methods from HillClimbingOptimizer)
"""

import numpy as np

from ..local_opt import HillClimbingOptimizer


class RandomAnnealingOptimizer(HillClimbingOptimizer):
    """Random Annealing with temperature-controlled step size.

    Unlike Simulated Annealing which uses temperature to control acceptance
    probability, Random Annealing uses temperature to control the step size
    (epsilon). This creates large initial exploration that gradually focuses
    to finer local search as the temperature decreases.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    The effective step size at each iteration is: epsilon * temp
    where temp starts at start_temp and decays by annealing_rate each iteration.

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
        Base step size for generating neighbors.
    distribution : str, default="normal"
        Distribution for step sizes: "normal", "laplace", or "logistic".
    n_neighbours : int, default=3
        Number of neighbors to evaluate before selecting the best.
    annealing_rate : float, default=0.98
        Temperature decay rate per iteration (temp *= annealing_rate).
    start_temp : float, default=10
        Initial temperature (step size multiplier).
    """

    name = "Random Annealing"
    _name_ = "random_annealing"
    __name__ = "RandomAnnealingOptimizer"

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
        annealing_rate=0.98,
        start_temp=10,
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
        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate new continuous values with temperature-scaled step size.

        Accesses via: self._pos_current, self._continuous_bounds
        """
        current = self._pos_current[self._continuous_mask]
        bounds = self._continuous_bounds

        ranges = bounds[:, 1] - bounds[:, 0]

        # Scale sigma by range, epsilon, AND temperature
        effective_epsilon = self.epsilon * self.temp
        sigmas = ranges * effective_epsilon

        noise_fn = self._DISTRIBUTIONS[self.distribution]
        noise = noise_fn(self._rng, sigmas, len(current))

        return current + noise

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Generate new categorical values with temperature-scaled switch probability.

        Accesses via: self._pos_current, self._categorical_sizes
        """
        current = self._pos_current[self._categorical_mask]
        n_categories = self._categorical_sizes

        n = len(current)

        # Switch probability scales with temperature
        effective_epsilon = self.epsilon * self.temp
        switch_prob = min(effective_epsilon, 1.0)  # Cap at 1.0

        switch_mask = self._rng.random(n) < switch_prob
        random_cats = np.floor(self._rng.random(n) * n_categories).astype(np.int64)

        return np.where(switch_mask, random_cats, current.astype(np.int64))

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Generate new discrete values with temperature-scaled step size.

        Accesses via: self._pos_current, self._discrete_bounds
        """
        current = self._pos_current[self._discrete_mask]
        bounds = self._discrete_bounds

        max_positions = bounds[:, 1]

        # Scale sigma by temperature
        effective_epsilon = self.epsilon * self.temp
        sigmas = max_positions * effective_epsilon
        sigmas = np.maximum(sigmas, 1e-10)

        noise_fn = self._DISTRIBUTIONS[self.distribution]
        noise = noise_fn(self._rng, sigmas, len(current))

        return current + noise

    def _evaluate(self, score_new):
        """Greedy evaluation with temperature decay.

        Uses the same n_neighbours greedy selection as HillClimbing,
        then decreases temperature for next iteration.
        """
        # Use parent's greedy n_neighbours logic
        super()._evaluate(score_new)

        # Decay temperature for next iteration
        self.temp *= self.annealing_rate
