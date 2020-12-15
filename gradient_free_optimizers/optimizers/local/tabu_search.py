# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from . import HillClimbingOptimizer
from ...search import Search


class TabuOptimizer(HillClimbingOptimizer, Search):
    def __init__(
        self,
        search_space,
        epsilon=0.05,
        distribution="normal",
        n_neighbours=3,
        tabu_factor=3,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space, epsilon, distribution, n_neighbours)

        self.tabus = []
        self.tabu_factor = tabu_factor
        self.rand_rest_p = rand_rest_p
        self.epsilon_mod = 1

    @HillClimbingOptimizer.track_nth_iter
    @HillClimbingOptimizer.random_restart
    def iterate(self):
        return self._move_climb(self.pos_current, epsilon_mod=self.epsilon_mod)

    def evaluate(self, score_new):
        super().evaluate(score_new)

        if score_new <= self.score_current:
            self.epsilon_mod = self.tabu_factor
        else:
            self.epsilon_mod = 1

