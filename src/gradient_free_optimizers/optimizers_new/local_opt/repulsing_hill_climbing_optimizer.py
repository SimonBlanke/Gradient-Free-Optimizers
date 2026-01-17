# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Repulsing Hill Climbing Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(inherits from HillClimbingOptimizer)
"""

from .hill_climbing_optimizer import HillClimbingOptimizer


class RepulsingHillClimbingOptimizer(HillClimbingOptimizer):
    """Hill Climbing with repulsion from visited positions.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    This optimizer remembers visited positions and adds a repulsion
    force to encourage exploration of unvisited regions.
    """

    name = "Repulsing Hill Climbing"
    _name_ = "repulsing_hill_climbing"
    __name__ = "RepulsingHillClimbingOptimizer"

    optimizer_type = "local"
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
        repulsion_factor=1.0,
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
        self.repulsion_factor = repulsion_factor

    def evaluate(self, score_new):
        """Evaluate with repulsion memory."""
        # TODO: Implement repulsion-based evaluation
        raise NotImplementedError("evaluate() not yet implemented")
