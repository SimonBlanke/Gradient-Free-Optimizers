# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._init_utils import get_default_initialize

from ..local_opt import HillClimbingOptimizer


class RandomAnnealingOptimizer(HillClimbingOptimizer):
    """Random annealing with decreasing step size.

    Similar to simulated annealing but without probabilistic acceptance.
    The step size (epsilon) is multiplied by a decaying temperature,
    causing large initial exploration that focuses over time.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    epsilon : float, default=0.03
        Base step size for generating neighbors.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors to evaluate.
    annealing_rate : float, default=0.98
        Temperature decay rate per iteration.
    start_temp : float, default=10
        Initial temperature (step size multiplier).
    """

    name = "Random Annealing"
    _name_ = "random_annealing"
    __name__ = "RandomAnnealingOptimizer"

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        annealing_rate=0.98,
        start_temp=10,
    ):
        if initialize is None:
            initialize = get_default_initialize()

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
        """Generate next position with temperature-scaled step size."""
        pos = self.move_climb(
            self.pos_current,
            epsilon=self.epsilon,
            distribution=self.distribution,
            epsilon_mod=self.temp,
        )
        self.temp = self.temp * self.annealing_rate

        return pos
