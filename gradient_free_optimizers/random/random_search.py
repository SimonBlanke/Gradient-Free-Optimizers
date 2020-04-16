# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_positioners = 1

    def _iterate(self, i, _cand_):
        if i < 1:
            self._init_iteration(_cand_)
        else:
            self.p_list[0].move_random(_cand_)
            self._optimizer_eval(_cand_, self.p_list[0])

            self._update_pos(_cand_, self.p_list[0])

        return _cand_

    def _init_iteration(self, _cand_):
        p = super()._init_base_positioner(_cand_)

        self._optimizer_eval(_cand_, p)
        self._update_pos(_cand_, p)

        return p
