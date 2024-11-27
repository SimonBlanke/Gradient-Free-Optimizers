# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .core_optimizer import CoreOptimizer


class BaseOptimizer(CoreOptimizer):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.optimizers = [self]

    def finish_initialization(self):
        self.search_state = "iter"

    def evaluate(self, score_new):
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.pos_current = self.pos_new

            self.score_best = score_new
            self.score_current = score_new
