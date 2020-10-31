# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class Individual(HillClimbingOptimizer):
    def __init__(self, search_space, rand_rest_p=0.03):
        super().__init__(search_space)
        self.rand_rest_p = rand_rest_p
