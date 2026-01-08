# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math

from ..core_optimizer import CoreOptimizer


def split(positions_l, population):
    div_int = math.ceil(len(positions_l) / population)
    dist_init_positions = []

    for nth_indiv in range(population):
        indiv_pos = []
        for nth_indiv_pos in range(div_int):
            idx = nth_indiv + nth_indiv_pos * population
            if idx < len(positions_l):
                indiv_pos.append(positions_l[idx])

        dist_init_positions.append(indiv_pos)

    return dist_init_positions


class BasePopulationOptimizer(CoreOptimizer):
    """Base class for population-based optimization algorithms.

    Manages a population of individual optimizers that explore the search
    space in parallel. Provides common functionality for creating populations,
    distributing initial positions, and tracking the best individual.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
        Strategy for generating initial positions distributed across population.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.

    Attributes
    ----------
    optimizers : list
        List of individual optimizer instances in the population.
    pop_sorted : list
        Population sorted by score (best first).
    """

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

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

        # Sort indices by score descending (pure Python)
        indexed = list(enumerate(scores_list))
        indexed.sort(key=lambda x: x[1], reverse=True)
        idx_sorted_ind = [i for i, _ in indexed]

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
            distributed_init_positions = split(
                self.init.init_positions_l, self.population
            )

            population = []
            for init_positions in distributed_init_positions:
                init_values = self.conv.positions2values(init_positions)
                init_paras = self.conv.values2paras(init_values)

                population.append(
                    Optimizer(
                        self.conv.search_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"warm_start": init_paras},
                        constraints=self.constraints,
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
