# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local_opt import HillClimbingOptimizer
from ...search import Search


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer, Search):
    name = "Random Restart Hill Climbing"

    def __init__(
        self,
        *args,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        n_iter_restart=10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.n_iter_restart = n_iter_restart

    @HillClimbingOptimizer.track_nth_iter
    @HillClimbingOptimizer.random_restart
    def iterate(self):
        notZero = self.nth_iter != 0
        modZero = self.nth_iter % self.n_iter_restart == 0

        if notZero and modZero:
            pos = self.move_random()
        else:
            pos = self._move_climb(self.pos_current)

        return pos
