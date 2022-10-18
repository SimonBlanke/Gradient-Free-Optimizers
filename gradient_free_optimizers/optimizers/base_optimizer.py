# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .core_optimizer import CoreOptimizer


class BaseOptimizer(CoreOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state = "init"

        self.optimizers = [self]

    def finish_initialization(self):
        self.search_state = "iter"

    def evaluate(self, score_new):
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.pos_current = self.pos_new

            self.score_best = score_new
            self.score_current = score_new
