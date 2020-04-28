# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomAnnealingOptimizer(HillClimbingOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.temp = self._opt_args_.start_temp

    def iterate(self, nth_iter):
        self._base_iterate(nth_iter)
        self._sort_()
        self._choose_next_pos()

        pos = self.p_current.move_climb(
            self.p_current.pos_current, epsilon_mod=self.temp / 10
        )

        self.temp = self.temp * self._opt_args_.annealing_rate

        return pos
