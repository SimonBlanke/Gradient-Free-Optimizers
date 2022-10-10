# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..core_optimizer import CoreOptimizer


def get_n_inits(initialize):
    n_inits = 0
    for key_ in initialize.keys():
        init_value = initialize[key_]
        if isinstance(init_value, int):
            n_inits += init_value
        else:
            n_inits += len(init_value)
    return n_inits


class BasePopulationOptimizer(CoreOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_times = []
        self.iter_times = []

        self.init_done = False

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter

    def sort_pop_best_score(self):
        scores_list = []
        for _p_ in self.optimizers:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        self.pop_sorted = [self.optimizers[i] for i in idx_sorted_ind]

    def _create_population(self, Optimizer):
        if isinstance(self.population, int):
            population = []
            for _ in range(self.population):
                population.append(
                    Optimizer(
                        self.conv.search_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"random": 1},
                    )
                )
        else:
            population = self.population

        n_inits = get_n_inits(self.initialize)
        diff_init = len(population) - n_inits

        if diff_init > 0:
            self.add_n_random_init_pos(diff_init)

        return population

    def finish_initialization(self):
        self.init_done = True
