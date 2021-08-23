# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search


class RandomSearchOptimizer(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        random_state=None,
    ):
        super().__init__(search_space, initialize, random_state)

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        return self.move_random()
