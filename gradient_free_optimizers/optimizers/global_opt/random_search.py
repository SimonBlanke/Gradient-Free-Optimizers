# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search


class RandomSearchOptimizer(BaseOptimizer, Search):
    name = "Random Search"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        return self.move_random()
