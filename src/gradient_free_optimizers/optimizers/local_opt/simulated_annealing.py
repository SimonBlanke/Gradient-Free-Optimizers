# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local_opt import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    name = "Simulated Annealing"
    _name_ = "simulated_annealing"
    __name__ = "SimulatedAnnealingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
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
        )

        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp

    def _p_accept_default(self):
        # the 'minus' is omitted, because we maximize a score
        return np.exp(self._exponent)

    def evaluate(self, score_new):
        StochasticHillClimbingOptimizer.evaluate(self, score_new)
        self.temp *= self.annealing_rate
