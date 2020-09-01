# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer
from ...search import Search


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer, Search):
    def __init__(self, search_space, n_iter_restart=10):
        super().__init__(search_space)
        self.n_iter_restart = n_iter_restart

    @HillClimbingOptimizer.iter_dec
    def iterate(self):
        notZero = self.nth_iter != 0
        modZero = self.nth_iter % self.n_iter_restart == 0

        if notZero and modZero:
            pos = self.move_random()
        else:
            pos = self._move_climb(self.pos_current)

        return pos
