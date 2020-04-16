# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from math import floor, ceil
import numpy as np
import random

from ..base_optimizer import BaseOptimizer
from ..local import HillClimbingPositioner


class EvolutionStrategyOptimizer(BaseOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_positioners = self._opt_args_.individuals

        self.n_mutations = floor(self.n_positioners * self._opt_args_.mutation_rate)
        self.n_crossovers = ceil(self.n_positioners * self._opt_args_.crossover_rate)

    def _init_individual(self, _cand_):
        _p_ = Individual(self._opt_args_)
        _p_.move_random(_cand_)

        return _p_

    def _mutate_individuals(self, _cand_, mutate_idx):
        p_list_mutate = [self.p_list[i] for i in mutate_idx]
        for _p_ in p_list_mutate:
            _p_.move_climb(_cand_, _p_.pos_new)

    def _crossover(self, _cand_, cross_idx, replace_idx):
        p_list_replace = [self.p_list[i] for i in replace_idx]
        for i, _p_ in enumerate(p_list_replace):
            j = i + 1
            if j == len(cross_idx):
                j = 0

            pos_new = self._cross_two_ind(
                [
                    [self.p_list[i] for i in cross_idx][i],
                    [self.p_list[i] for i in cross_idx][j],
                ]
            )

            _p_.pos_new = pos_new

    def _cross_two_ind(self, p_list):
        pos_new = []

        for pos1, pos2 in zip(p_list[0].pos_new, p_list[1].pos_new):
            rand = random.randint(0, 1)
            if rand == 0:
                pos_new.append(pos1)
            else:
                pos_new.append(pos2)

        return np.array(pos_new)

    def _move_positioners(self, _cand_):
        idx_sorted_ind = self._rank_individuals()
        mutate_idx, cross_idx, replace_idx = self._select_individuals(idx_sorted_ind)

        self._crossover(_cand_, cross_idx, replace_idx)
        self._mutate_individuals(_cand_, mutate_idx)

    def _rank_individuals(self):
        scores_list = []
        for _p_ in self.p_list:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return idx_sorted_ind

    def _select_individuals(self, index_best):
        mutate_idx = index_best[: self.n_mutations]
        cross_idx = index_best[: self.n_crossovers]

        n = self.n_crossovers
        replace_idx = index_best[-n:]

        return mutate_idx, cross_idx, replace_idx

    def _iterate(self, i, _cand_):
        _p_current = self.p_list[i % self.n_positioners]

        self._move_positioners(_cand_)
        self._optimizer_eval(_cand_, _p_current)
        self._update_pos(_cand_, _p_current)

        return _cand_

    def _init_iteration(self, _cand_):
        p = self._init_individual(_cand_)

        self._optimizer_eval(_cand_, p)
        self._update_pos(_cand_, p)

        return p


class Individual(HillClimbingPositioner):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)
