# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer


class EvolutionaryAlgorithmOptimizer(BasePopulationOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def discrete_recombination(self, parent_pos_l, crossover_rates=None):
        n_parents = len(parent_pos_l)
        size = parent_pos_l[0].size

        choice = []
        for _ in range(size):
            choices = list(range(n_parents))
            choice.append(np.random.choice(choices, p=crossover_rates))

        return np.choose(choice, parent_pos_l)
