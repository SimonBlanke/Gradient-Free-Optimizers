# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local_opt import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    name = "Random Restart Hill Climbing"
    _name_ = "random_restart_hill_climbing"
    __name__ = "RandomRestartHillClimbingOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        *args,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        n_iter_restart=10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.n_iter_restart = n_iter_restart

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        notZero = self.nth_trial != 0
        modZero = self.nth_trial % self.n_iter_restart == 0

        if notZero and modZero:
            return self.move_random()
        else:
            return self.move_climb(self.pos_current)
