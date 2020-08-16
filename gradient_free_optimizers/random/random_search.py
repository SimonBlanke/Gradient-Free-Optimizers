# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, space_dim):
        super().__init__(space_dim)

    def iterate(self, nth_iter):
        return self.move_random()

