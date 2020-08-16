# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class BasePopulationOptimizer:
    def __init__(self, space_dim):
        self.space_dim = space_dim

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter
