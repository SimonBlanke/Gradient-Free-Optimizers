# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..core_optimizer import CoreOptimizer


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
            pop_size = self.population
        else:
            pop_size = len(self.population)
        diff_init = pop_size - self.init.n_inits

        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        if isinstance(self.population, int):
            population = []
            for init_position in self.init.init_positions_l:
                init_value = self.conv.position2value(init_position)
                init_para = self.conv.value2para(init_value)

                population.append(
                    Optimizer(
                        self.conv.search_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"warm_start": [init_para]},
                    )
                )
        else:
            population = self.population

        return population

    @CoreOptimizer.track_new_score
    def evaluate_init(self, score_new):
        self.p_current.evaluate_init(score_new)

    def finish_initialization(self):
        self.search_state = "iter"
