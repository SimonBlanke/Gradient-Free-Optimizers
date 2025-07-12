# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer


def max_list_idx(list_):
    max_item = max(list_)
    max_item_idx = [i for i, j in enumerate(list_) if j == max_item]
    return max_item_idx[-1:][0]


class HillClimbingOptimizer(BaseOptimizer):
    name = "Hill Climbing"
    _name_ = "hill_climbing"
    __name__ = "HillClimbingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
    ):

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours

    @BaseOptimizer.track_new_pos
    @BaseOptimizer.random_iteration
    def iterate(self):
        return self.move_climb(
            self.pos_current,
            epsilon=self.epsilon,
            distribution=self.distribution,
        )

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:
            return

        modZero = self.nth_trial % self.n_neighbours == 0
        if modZero:
            score_new_list_temp = self.scores_valid[-self.n_neighbours :]
            pos_new_list_temp = self.positions_valid[-self.n_neighbours :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)
