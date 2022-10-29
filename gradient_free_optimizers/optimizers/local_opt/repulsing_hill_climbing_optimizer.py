# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from . import HillClimbingOptimizer
from ...search import Search


class RepulsingHillClimbingOptimizer(HillClimbingOptimizer, Search):
    name = "Repulsing Hill Climbing"
    _name_ = "repulsing_hill_climbing"
    __name__ = "RepulsingHillClimbingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(self, *args, repulsion_factor=5, **kwargs):
        super().__init__(*args, **kwargs)

        self.tabus = []
        self.repulsion_factor = repulsion_factor
        self.epsilon_mod = 1

    @HillClimbingOptimizer.track_new_pos
    def iterate(self):
        return self._move_climb(self.pos_current, epsilon_mod=self.epsilon_mod)

    def evaluate(self, score_new):
        super().evaluate(score_new)

        if score_new <= self.score_current:
            self.epsilon_mod = self.repulsion_factor
        else:
            self.epsilon_mod = 1
