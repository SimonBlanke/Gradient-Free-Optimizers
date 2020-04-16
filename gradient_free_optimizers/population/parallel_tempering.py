# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ..local import HillClimbingPositioner, SimulatedAnnealingOptimizer


class ParallelTemperingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_iter_swap = self._opt_args_.n_iter_swap
        self.n_positioners = len(self._opt_args_.system_temperatures)

    def _init_annealer(self, _cand_):
        temp = self._opt_args_.system_temperatures[self.i]
        _p_ = System(self._opt_args_, temp=temp)

        _p_.pos_new = _cand_._space_.get_random_pos()

        return _p_

    def _swap_pos(self, _cand_):
        _p_list_temp = self.p_list[:]

        for _p1_ in self.p_list:
            rand = random.uniform(0, 1)
            _p2_ = np.random.choice(_p_list_temp)

            p_accept = self._accept_swap(_p1_, _p2_)
            if p_accept > rand:
                temp_temp = _p1_.temp  # haha!
                _p1_.temp = _p2_.temp
                _p2_.temp = temp_temp

    def _accept_swap(self, _p1_, _p2_):
        denom = _p1_.score_current + _p2_.score_current

        if denom == 0:
            return 100
        elif abs(denom) == np.inf:
            return 0
        else:
            score_diff_norm = (_p1_.score_current - _p2_.score_current) / denom

            temp = (1 / _p1_.temp) - (1 / _p2_.temp)
            return np.exp(score_diff_norm * temp)

    def _anneal_system(self, _cand_, _p_):
        self._p_ = _p_
        super()._iterate(0, _cand_)

    def _iterate(self, i, _cand_):
        _p_current = self.p_list[i % len(self.p_list)]

        self._anneal_system(_cand_, _p_current)

        if self.n_iter_swap != 0 and i % self.n_iter_swap == 0:
            self._swap_pos(_cand_)

        return _cand_

    def _init_iteration(self, _cand_):
        p = self._init_annealer(_cand_)

        self._optimizer_eval(_cand_, p)
        self._update_pos(_cand_, p)

        return p

    def _finish_search(self):
        self._pbar_.close_p_bar()


class System(HillClimbingPositioner):
    def __init__(self, _opt_args_, temp):
        super().__init__(_opt_args_)
        self.temp = temp
