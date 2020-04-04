# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer
from ..base_positioner import BasePositioner


class HillClimbingOptimizer(BaseOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_positioners = 1

    def _hill_climb_iter(self, i, _cand_):
        score_new = -np.inf
        pos_new = None

        self.p_list[0].move_climb(_cand_, self.p_list[0].pos_current)
        self._optimizer_eval(_cand_, self.p_list[0])

        if self.p_list[0].score_new > score_new:
            score_new = self.p_list[0].score_new
            pos_new = self.p_list[0].pos_new

        if i % self._opt_args_.n_neighbours == 0:
            self.p_list[0].pos_new = pos_new
            self.p_list[0].score_new = score_new

            self._update_pos(_cand_, self.p_list[0])

    def _iterate(self, i, _cand_):
        self._hill_climb_iter(i, _cand_)

    def _init_iteration(self, _cand_):
        p = super()._init_base_positioner(_cand_, positioner=HillClimbingPositioner)

        self._optimizer_eval(_cand_, p)
        self._update_pos(_cand_, p)

        return p


class HillClimbingPositioner(BasePositioner):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)

        self.epsilon = _opt_args_.epsilon
        self.distribution = _opt_args_.distribution
