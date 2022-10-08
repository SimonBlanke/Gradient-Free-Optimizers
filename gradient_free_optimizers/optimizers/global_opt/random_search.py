# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from ...search import Search


class RandomSearchOptimizer(BaseOptimizer, Search):
    name = "Random Search"
    _name_ = "random_search"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseOptimizer.track_new_pos
    def iterate(self):
        return self.move_random()
    @BaseOptimizer.track_new_score

    def evaluate(self, score_new):
        return super().evaluate(score_new)
