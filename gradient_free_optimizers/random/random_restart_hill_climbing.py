# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)

    def iterate(self, nth_iter):
        self._base_iterate(nth_iter)
        self._sort_()
        self._choose_next_pos()

        notZero = self._opt_args_.n_iter_restart != 0
        modZero = nth_iter % self._opt_args_.n_iter_restart == 0

        if notZero and modZero:
            pos = self.p_current.move_random()
        else:
            pos = self.p_current.move_climb(self.p_current.pos_current)

        return pos
