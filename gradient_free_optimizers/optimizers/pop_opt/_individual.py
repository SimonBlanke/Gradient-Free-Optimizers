# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local_opt import HillClimbingOptimizer


class Individual(HillClimbingOptimizer):
    def __init__(self, *args, rand_rest_p=0.03, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand_rest_p = rand_rest_p
