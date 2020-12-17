# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer
from ...search import Search


class RandomAnnealingOptimizer(HillClimbingOptimizer, Search):
    def __init__(
        self,
        search_space,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        annealing_rate=0.97,
        start_temp=1,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space)
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp
        self.rand_rest_p = rand_rest_p

    @HillClimbingOptimizer.track_nth_iter
    @HillClimbingOptimizer.random_restart
    def iterate(self):
        pos = self._move_climb(self.pos_current, epsilon_mod=self.temp * 10)
        self.temp = self.temp * self.annealing_rate

        return pos
