# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math

from gradient_free_optimizers._array_backend import random as np_random

from ..base_optimizer import BaseOptimizer
from ..local_opt import HillClimbingOptimizer


class Individual(HillClimbingOptimizer):
    def __init__(self, *args, rand_rest_p=0.03, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand_rest_p = rand_rest_p

        # Initialize sigma self-adaptation (Schwefel, 1981)
        self.sigma = self.epsilon
        self.sigma_new = self.sigma

        # Learning rate: tau = 1/sqrt(n) where n = number of dimensions
        n_dimensions = len(self.search_space)
        self.tau = 1.0 / math.sqrt(n_dimensions)

        # Bounds to prevent sigma collapse or divergence
        self.sigma_min = 0.001
        self.sigma_max = 0.5

    @BaseOptimizer.track_new_pos
    @BaseOptimizer.random_iteration
    def iterate(self):
        # Mutate sigma (log-normal distribution)
        self.sigma_new = self.sigma * math.exp(self.tau * np_random.normal())
        self.sigma_new = max(self.sigma_min, min(self.sigma_new, self.sigma_max))

        return self.move_climb_typed(
            self.pos_current,
            epsilon=self.sigma_new,
            distribution=self.distribution,
        )

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)

        # Adopt new sigma only if mutation was successful
        if score_new > self.score_current:
            self.sigma = self.sigma_new

        self._eval2current(self.pos_new, score_new)
        self._eval2best(self.pos_new, score_new)
