# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ..local import HillClimbingPositioner, SimulatedAnnealingOptimizer


class ParallelTemperingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.n_positioners = len(self._opt_args_.system_temperatures)

    def init_pos(self, nth_init):
        temp = self._opt_args_.system_temperatures[nth_init]
        pos_new = self._base_init_pos(
            nth_init, System(self.space_dim, self._opt_args_, temp)
        )

        return pos_new

    def _swap_pos(self):
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

    def evaluate(self, score_new):
        super().evaluate(score_new)

        notZero = self._opt_args_.n_iter_swap != 0
        modZero = self.nth_iter % self._opt_args_.n_iter_swap == 0

        if notZero and modZero:
            self._swap_pos()


class System(HillClimbingPositioner):
    def __init__(self, space_dim, _opt_args_, temp):
        super().__init__(space_dim, _opt_args_)
        self.temp = temp
