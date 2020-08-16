# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomAnnealingOptimizer(HillClimbingOptimizer):
    def __init__(self, space_dim, annealing_rate=0.99, start_temp=100):
        super().__init__(space_dim)
        self.annealing_rate = annealing_rate
        self.temp = start_temp

    def iterate(self):
        pos = self._move_climb(self.pos_current, epsilon_mod=self.temp / 10)
        self.temp = self.temp * self.annealing_rate

        return pos
