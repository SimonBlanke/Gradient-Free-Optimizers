# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer
from ...search import Search


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer, Search):
    def __init__(self, search_space, n_iter_restart=10, **kwargs):
        super().__init__(search_space, **kwargs)
        self.n_iter_restart = n_iter_restart

    def iterate(self):
        notZero = self.n_iter_restart != 0
        modZero = len(self.pos_new) % self.n_iter_restart == 0

        if notZero and modZero:
            pos = self.move_random()
        else:
            pos = self._move_climb(self.pos_current)

        return pos
