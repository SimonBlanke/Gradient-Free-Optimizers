# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search


class RandomSearchOptimizer(BaseOptimizer, Search):
    def __init__(self, search_space, **kwargs):
        super().__init__(search_space)

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        return self.move_random()

