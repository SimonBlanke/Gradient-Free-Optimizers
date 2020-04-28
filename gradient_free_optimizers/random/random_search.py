# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer
from ..base_positioner import BasePositioner


class RandomSearchOptimizer(HillClimbingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)

    def init_pos(self, init_position):
        pos_new = self._base_init_pos(
            init_position, RandomSearchPositioner(self.space_dim, self._opt_args_)
        )

        return pos_new

    def iterate(self, nth_iter):
        self._base_iterate(nth_iter)
        self._sort_()
        self._choose_next_pos()

        pos = self.p_current.move_random()

        return pos


class RandomSearchPositioner(BasePositioner):
    def __init__(self, space_dim, _opt_args_):
        super().__init__(space_dim, _opt_args_)

        self.epsilon = _opt_args_.epsilon
        self.distribution = _opt_args_.distribution
