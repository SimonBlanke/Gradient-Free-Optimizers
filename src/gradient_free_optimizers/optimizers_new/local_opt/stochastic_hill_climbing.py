# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Stochastic Hill Climbing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(inherits iteration methods from HillClimbingOptimizer)
"""

import math

import numpy as np

from .hill_climbing_optimizer import HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HillClimbingOptimizer):
    """Stochastic Hill Climbing with probabilistic acceptance of worse solutions.

    Unlike pure hill climbing, this optimizer may accept worse solutions with a
    probability based on the score difference. This helps escape local optima
    while still preferring uphill moves.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    The acceptance probability for worse solutions uses a sigmoid function:
        p = p_accept * 2 / (1 + exp(-normalized_energy / temp))

    Where normalized_energy = (score_new - score_current) / (score_new + score_current)

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
    p_accept : float, default=0.5
        Base probability for accepting worse solutions.
    """

    name = "Stochastic Hill Climbing"
    _name_ = "stochastic_hill_climbing"
    __name__ = "StochasticHillClimbingOptimizer"

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
        p_accept=0.5,
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
        self.p_accept = p_accept
        self.temp = 1  # Constant temperature for stochastic hill climbing

        # Transition tracking (for diagnostics)
        self.n_transitions = 0
        self.n_considered_transitions = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # ACCEPTANCE PROBABILITY CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def _normalized_energy_state(self) -> float:
        """Calculate normalized energy difference between new and current scores.

        Returns a value in approximately [-1, 1] that represents how much
        worse (negative) or better (positive) the new score is compared
        to the current score, normalized by their sum.
        """
        denom = self.score_current + self.score_new

        if denom == 0:
            return 1
        elif math.isinf(abs(denom)):
            return 0
        else:
            return (self.score_new - self.score_current) / denom

    @property
    def _exponent(self) -> float:
        """Calculate the exponent for the acceptance probability."""
        if self.temp == 0:
            return -math.inf
        else:
            return self._normalized_energy_state / self.temp

    def _p_accept_default(self) -> float:
        """Calculate the acceptance probability for a worse solution.

        Uses a sigmoid function that maps the normalized energy difference
        to a probability, scaled by the base p_accept parameter.

        Returns
        -------
        float
            Probability of accepting the current worse solution.
        """
        try:
            exp_val = math.exp(self._exponent)
        except OverflowError:
            exp_val = math.inf
        return self.p_accept * 2 / (1 + exp_val)

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATE: Stochastic acceptance of worse solutions
    # ═══════════════════════════════════════════════════════════════════════════

    def _evaluate(self, score_new):
        """Evaluate with stochastic acceptance of worse solutions.

        If the new score is better than current, use standard hill climbing
        behavior (greedy selection after n_neighbours trials).

        If the new score is worse, accept it with a probability based on
        the score difference. This allows escaping local optima.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Store score_new for property access
        self.score_new = score_new

        # If score is better or equal, use standard hill climbing logic
        if score_new > self.score_current:
            super()._evaluate(score_new)
        else:
            # Score is worse - consider stochastic acceptance
            self.n_considered_transitions += 1
            p_accept = self._p_accept_default()

            if self._rng.random() < p_accept:
                # Accept the worse solution (this is what n_transitions counts)
                self.n_transitions += 1
                self._update_current(self.pos_new, score_new)

            # Always update best (in case this is somehow better than global best)
            self._update_best(self.pos_new, score_new)
