# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, space_dim, n_iter_restart=10):
        super().__init__(space_dim)
        self.n_iter_restart = n_iter_restart

    def iterate(self, nth_iter):
        notZero = self.n_iter_restart != 0
        modZero = nth_iter % self.n_iter_restart == 0

        if notZero and modZero:
            pos = self.move_random()
        else:
            pos = self._move_climb(self.pos_current)

        return pos
