# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    name = "Random Search"
    _name_ = "random_search"
    __name__ = "RandomSearchOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

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

    @BaseOptimizer.track_new_pos
    def iterate(self):
        return self.move_random()

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        return super().evaluate(score_new)
