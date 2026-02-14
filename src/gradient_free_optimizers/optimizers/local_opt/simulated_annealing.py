# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Simulated Annealing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL.
Inherits iteration methods from HillClimbingOptimizer via StochasticHillClimbing.
"""

import math

from .stochastic_hill_climbing import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    """Simulated Annealing optimizer inspired by metallurgical annealing.

    Uses a temperature parameter that decreases over time, controlling the
    probability of accepting worse solutions. High temperature allows more
    exploration; low temperature focuses on exploitation.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    The acceptance probability follows the Metropolis criterion:
        p = exp(normalized_energy / temp)

    Where normalized_energy = (score_new - score_current) / (score_new + score_current)

    The temperature decreases each iteration: temp *= annealing_rate

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
    annealing_rate : float, default=0.97
        Temperature decay rate per iteration (temp *= annealing_rate).
    start_temp : float, default=1
        Initial temperature value.
    """

    name = "Simulated Annealing"
    _name_ = "simulated_annealing"
    __name__ = "SimulatedAnnealingOptimizer"

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
        annealing_rate=0.97,
        start_temp=1,
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
            # Note: p_accept is not used in SA, we use pure Metropolis criterion
        )
        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp

    def _p_accept_default(self) -> float:
        """Calculate the Metropolis acceptance probability.

        For simulated annealing, we use the classic Metropolis criterion:
            p = exp(delta / temp)

        where delta = (score_new - score_current) / (score_new + score_current)

        Note: The minus sign is omitted because we maximize (not minimize) scores.

        Returns
        -------
        float
            Probability of accepting the current solution.
        """
        try:
            return math.exp(self._exponent)
        except OverflowError:
            return math.inf

    def _on_evaluate(self, score_new):
        """Evaluate with Metropolis acceptance criterion and temperature decay.

        After the stochastic acceptance decision, the temperature is reduced
        according to the annealing schedule: temp *= annealing_rate

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Use parent's stochastic acceptance logic
        super()._on_evaluate(score_new)

        # Decrease temperature (annealing schedule)
        self.temp *= self.annealing_rate
