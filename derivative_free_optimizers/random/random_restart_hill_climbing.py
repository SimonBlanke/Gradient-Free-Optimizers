# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_iter_restart = self._opt_args_.n_iter_restart

    def _iterate(self, i, _cand_):
        self._hill_climb_iter(i, _cand_)

        if self.n_iter_restart != 0 and i % self.n_iter_restart == 0:
            self.p_list[0].move_random(_cand_)

        return _cand_
