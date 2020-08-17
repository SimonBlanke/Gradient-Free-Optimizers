# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, space_dim, n_iter_restart=10, **kwargs):
        super().__init__(space_dim, **kwargs)
        self.n_iter_restart = n_iter_restart

    def iterate(self):
        notZero = self.n_iter_restart != 0
        modZero = len(self.pos_new) % self.n_iter_restart == 0

        if notZero and modZero:
            pos = self.move_random()
        else:
            pos = self._move_climb(self.pos_current)

        return pos
