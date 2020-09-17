# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer
from ...search import Search


class RandomAnnealingOptimizer(HillClimbingOptimizer, Search):
    def __init__(self, search_space, annealing_rate=0.975, start_temp=1, **kwargs):
        super().__init__(search_space)
        self.annealing_rate = annealing_rate
        self.temp = start_temp

    def iterate(self):
        pos = self._move_climb(self.pos_current, epsilon_mod=self.temp * 10)
        self.temp = self.temp * self.annealing_rate

        return pos
