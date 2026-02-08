# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from gradient_free_optimizers._init_utils import get_default_initialize

from ..local_opt import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    """Hill climbing with periodic random restarts.

    Combines local hill climbing with global exploration by periodically
    restarting from a random position. This helps escape local optima
    while still exploiting local structure.

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
        Step size for hill climbing.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors to evaluate.
    n_iter_restart : int, default=10
        Number of iterations between random restarts.
    """

    name = "Random Restart Hill Climbing"
    _name_ = "random_restart_hill_climbing"
    __name__ = "RandomRestartHillClimbingOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

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
        n_iter_restart=10,
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
        self.n_iter_restart = n_iter_restart

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        """Hill climb or random restart based on iteration count."""
        notZero = self.nth_trial != 0
        modZero = self.nth_trial % self.n_iter_restart == 0

        if notZero and modZero:
            return self.move_random()
        else:
            return self.move_climb(
                self.pos_current,
                epsilon=self.epsilon,
                distribution=self.distribution,
            )
