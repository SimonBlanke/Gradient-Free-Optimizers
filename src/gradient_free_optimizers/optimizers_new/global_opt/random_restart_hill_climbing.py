# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Random Restart Hill Climbing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(inherits from HillClimbingOptimizer)
"""

from ..local_opt import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    """Hill Climbing with periodic random restarts.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    Periodically restarts from a random position to escape local optima.
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
        n_iter_restart=10,
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
        )
        self.n_iter_restart = n_iter_restart

    def evaluate(self, score_new):
        """Evaluate with restart logic."""
        # TODO: Implement restart logic
        raise NotImplementedError("evaluate() not yet implemented")
