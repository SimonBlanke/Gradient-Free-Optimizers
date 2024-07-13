# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer


class EvolutionaryAlgorithmOptimizer(BasePopulationOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def discrete_recombination(self, parent_pos_l):
        n_arrays = len(parent_pos_l)
        size = parent_pos_l[0].size

        if random.choice([True, False]):
            choice = [True, False]
        else:
            choice = [False, True]
        if size > 2:
            add_choice = np.random.randint(n_arrays, size=size - 2).astype(bool)
            choice += list(add_choice)
        return np.choose(choice, parent_pos_l)
