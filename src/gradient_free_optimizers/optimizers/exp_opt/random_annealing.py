# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local_opt import HillClimbingOptimizer


class RandomAnnealingOptimizer(HillClimbingOptimizer):
    name = "Random Annealing"
    _name_ = "random_annealing"
    __name__ = "RandomAnnealingOptimizer"

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
        annealing_rate=0.98,
        start_temp=10,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
        )
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        pos = self.move_climb(
            self.pos_current,
            epsilon=self.epsilon,
            distribution=self.distribution,
            epsilon_mod=self.temp,
        )
        self.temp = self.temp * self.annealing_rate

        return pos
