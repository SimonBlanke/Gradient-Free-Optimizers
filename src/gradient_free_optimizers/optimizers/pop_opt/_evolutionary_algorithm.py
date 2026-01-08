# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random

from gradient_free_optimizers._array_backend import array, random as np_random

from .base_population_optimizer import BasePopulationOptimizer


class EvolutionaryAlgorithmOptimizer(BasePopulationOptimizer):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

    def discrete_recombination(self, parent_pos_l, crossover_rates=None):
        n_parents = len(parent_pos_l)
        size = parent_pos_l[0].size

        # Select parent index for each position (replaces np.random.choice)
        choice = []
        for _ in range(size):
            choices = list(range(n_parents))
            choice.append(np_random.choice(choices, p=crossover_rates))

        # Build result by selecting from parents (replaces np.choose)
        result = []
        for i, parent_idx in enumerate(choice):
            result.append(parent_pos_l[parent_idx][i])
        return array(result)
